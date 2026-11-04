from __future__ import annotations

import argparse
import json
import platform
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image, __version__ as pillow_version
from torch.nn import functional as F
from torchvision import __version__ as torchvision_version
from torchvision.models import ResNet50_Weights, resnet50

from .provenance import RESNET50_SHA256, verify_weights


def extract(archive, output, weights_path):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.IMAGENET1K_V2
    if weights_path is None:
        weights_path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
        if not weights_path.exists():
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(weights.url, str(weights_path), RESNET50_SHA256)
    verify_weights(weights_path)
    model = resnet50()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model = model.eval().to(device)
    transform = weights.transforms(resize_size=224, crop_size=224)
    # Fixed projection preserves distances without fitting on evaluation images.
    rng = np.random.default_rng(2026)
    projection = torch.tensor(
        rng.normal(0, 1 / np.sqrt(192), (1792, 192)),
        dtype=torch.float32, device=device,
    )
    members, embeddings, patches = [], [], []
    with ZipFile(archive) as source, torch.inference_mode():
        members = sorted(n for n in source.namelist() if n.endswith(".png")
                         and not n.startswith("__MACOSX/"))
        for start in range(0, len(members), 16):
            batch = []
            for name in members[start:start + 16]:
                with Image.open(BytesIO(source.read(name))) as image:
                    batch.append(transform(image.convert("RGB").resize(
                        (224, 224), Image.Resampling.BILINEAR)))
            x = torch.stack(batch).to(device)
            x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            layer1 = model.layer1(x)
            layer2 = model.layer2(layer1)
            layer3 = model.layer3(layer2)
            layer4 = model.layer4(layer3)
            blocks = [F.normalize(layer.mean((2, 3)), dim=1) for layer in
                      (layer1, layer2, layer3, layer4)]
            embeddings.append(torch.cat(blocks, dim=1).cpu().numpy())
            if archive.stem == "Image_Anomaly_Detection":
                blocks = [
                    F.normalize(F.adaptive_avg_pool2d(layer, 14), dim=1)
                    for layer in (layer1, layer2)
                ]
                # Layer 3 contributes wider context to each local descriptor.
                context = F.normalize(layer3, dim=1)
                local = torch.cat([*blocks, context], dim=1).permute(0, 2, 3, 1)
                patches.append((local @ projection).flatten(1, 2).cpu().numpy())
    metadata = dict(
        weights_sha256=RESNET50_SHA256, python=platform.python_version(),
        torch=str(torch.__version__), torchvision=torchvision_version,
        numpy=np.__version__, pillow=pillow_version, device=str(device), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    payload = dict(names=np.array(members), embeddings=np.concatenate(embeddings),
                   metadata=json.dumps(metadata))
    if patches:
        payload["patches"] = np.concatenate(patches)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as work:
        temporary = Path(work) / "features.npz"
        np.savez(temporary, **payload)
        temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    extract(args.archive, args.output, args.weights)

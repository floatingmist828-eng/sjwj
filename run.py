import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from dm2026 import anomaly, cluster, weather
from dm2026.common import write_json
from dm2026.provenance import RESNET50_SHA256, file_sha256, verify_weights

TASKS = ("cluster", "image", "weather", "thyroid", "cross-modal")
ROOT = Path(__file__).resolve().parent
INPUTS = {
    "cluster": ("Cluster.zip",),
    "image": ("Image_Anomaly_Detection.zip",),
    "weather": ("weather.csv",),
    "thyroid": ("thyroid.zip",),
    "cross-modal": ("Image_Anomaly_Detection.zip", "thyroid.zip"),
}


def load_vision(archive, args, archive_digest):
    extractor_digest = file_sha256(ROOT / "dm2026" / "vision.py")
    key = hashlib.sha256(f"{archive_digest}:{extractor_digest}:{RESNET50_SHA256}".encode())
    cache = args.cache / f"{archive.stem}-{key.hexdigest()[:16]}.npz"
    cache_hit = cache.exists()
    if not cache_hit:
        interpreter = Path(args.vision_python)
        command = [str(interpreter.resolve()) if interpreter.is_file() else args.vision_python,
                   "-m", "dm2026.vision",
                   str(archive.resolve()), str(cache.resolve())]
        if args.weights:
            command.extend(["--weights", str(args.weights.resolve())])
        print(f"Extracting {archive.name} features", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    with np.load(cache, allow_pickle=False) as features:
        arrays = {name: features[name] for name in features.files if name != "metadata"}
        metadata = json.loads(features["metadata"].item())
    return arrays, dict(**metadata, cache_key=key.hexdigest(), cache_hit=cache_hit)


def main():
    parser = argparse.ArgumentParser(description="DM 2026 coursework experiments")
    parser.add_argument("task", choices=("all", *TASKS), nargs="?", default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "DM_2026_Dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--vision-python", default=sys.executable)
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    input_names = dict.fromkeys(name for task in tasks for name in INPUTS[task])
    input_hashes = {name: file_sha256(args.data / name) for name in input_names}
    if args.weights and any(name.endswith(".zip") and name != "thyroid.zip" for name in input_names):
        verify_weights(args.weights)
    paths = {task: args.output / task for task in tasks}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    visual_runtime = {}
    with threadpool_limits(limits=4):
        if "cluster" in tasks:
            archive = args.data / "Cluster.zip"
            features, visual_runtime[archive.name] = load_vision(
                archive, args, input_hashes[archive.name])
            cluster.run(archive, features, paths["cluster"])
            print(f"cluster: {paths['cluster']}")
        if "image" in tasks or "cross-modal" in tasks:
            archive = args.data / "Image_Anomaly_Detection.zip"
            features, visual_runtime[archive.name] = load_vision(
                archive, args, input_hashes[archive.name])
            image = anomaly.image_runs(archive, features, paths.get("image"), "image" in tasks)
            if "image" in tasks:
                print(f"image: {paths['image']}")
        if "weather" in tasks:
            weather.run(args.data / "weather.csv", paths["weather"])
            print(f"weather: {paths['weather']}")
        if "thyroid" in tasks or "cross-modal" in tasks:
            thyroid = anomaly.thyroid_run(args.data / "thyroid.zip", paths.get("thyroid"),
                                          "thyroid" in tasks)
            if "thyroid" in tasks:
                print(f"thyroid: {paths['thyroid']}")
        if "cross-modal" in tasks:
            anomaly.cross_modal(paths["cross-modal"], {**image, "thyroid": thyroid})
            print(f"cross-modal: {paths['cross-modal']}")
    sources = [ROOT / "run.py", ROOT / "requirements.txt", *sorted((ROOT / "dm2026").glob("*.py"))]
    write_json(args.output / f"run_{args.task}.json", dict(
        tasks=tasks, command=[sys.executable, *sys.argv], platform=platform.platform(),
        python=platform.python_version(), seed=2026, thread_limit=4,
        packages={name: version(name) for name in
                  ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "Pillow", "threadpoolctl")},
        input_sha256=input_hashes, vision=visual_runtime,
        source_sha256={path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sources}))


if __name__ == "__main__":
    main()

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from dm2026 import anomaly, cluster, weather

TASKS = ("cluster", "image", "weather", "thyroid", "cross-modal")
ROOT = Path(__file__).resolve().parent


def load_vision(archive, args):
    script = ROOT / "dm2026" / "vision.py"
    metadata = archive.stat()
    key = hashlib.sha256(script.read_bytes() +
                         f"{archive.resolve()}:{metadata.st_size}:{metadata.st_mtime_ns}".encode())
    cache = args.cache / f"{archive.stem}-{key.hexdigest()[:16]}.npz"
    if not cache.exists():
        command = [args.vision_python, str(script), str(archive), str(cache)]
        if args.weights:
            command.extend(["--weights", str(args.weights)])
        print(f"Extracting {archive.name} features", flush=True)
        subprocess.run(command, check=True)
    with np.load(cache, allow_pickle=False) as features:
        return dict(features)


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
    paths = {task: args.output / task for task in tasks}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        if "cluster" in tasks:
            archive = args.data / "Cluster.zip"
            cluster.run(archive, load_vision(archive, args), paths["cluster"])
            print(f"cluster: {paths['cluster']}")
        if "image" in tasks or "cross-modal" in tasks:
            archive = args.data / "Image_Anomaly_Detection.zip"
            image = anomaly.image_runs(archive, load_vision(archive, args),
                                       paths.get("image"), "image" in tasks)
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


if __name__ == "__main__":
    main()

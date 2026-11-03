from io import BytesIO
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from .common import binary_outputs, predictions, read_labels, write_json

SEED = 2026
ALPHA = 0.05


class NormalMemory:
    """Score bags of descriptors against a bank of normal descriptors."""

    def __init__(self, reference):
        self.dimension = reference.shape[-1]
        self.bag_size = reference.shape[1]
        bank = reference.reshape(-1, self.dimension)
        if len(bank) > 8192:
            rng = np.random.default_rng(SEED)
            bank = bank[rng.choice(len(bank), 8192, replace=False)]
        self.index = NearestNeighbors(n_neighbors=5, algorithm="auto").fit(bank)
        self.radii = self.index.kneighbors()[0].mean(axis=1)
        self.radius_floor = max(float(np.quantile(self.radii, 0.10)), 1e-8)

    def distances(self, bags):
        queries = bags.reshape(-1, self.dimension)
        distances, neighbors = self.index.kneighbors(queries)
        raw = distances.mean(axis=1)
        local = np.maximum(self.radii[neighbors].mean(axis=1), self.radius_floor)
        return raw.reshape(-1, self.bag_size), (raw / np.sqrt(local)).reshape(-1, self.bag_size)

    def aggregate(self, distances):
        count = max(1, int(np.ceil(self.bag_size * 0.05)))
        return np.sort(distances, axis=1)[:, -count:].mean(axis=1)


def normal_split(n):
    ids = np.random.default_rng(SEED).permutation(n)
    cutoff = n - int(np.ceil(n * 0.25))
    return ids[:cutoff], ids[cutoff:]


def conformal_p(calibration, scores):
    ranks = np.searchsorted(np.sort(calibration), scores, side="left")
    return (len(calibration) - ranks + 1) / (len(calibration) + 1)


def calibrated_frame(ids, y, calibration, scores):
    p = conformal_p(calibration, scores)
    return predictions(ids, y, scores, np.inf, p_normal=p).assign(
        prediction=(p <= ALPHA).astype(int))


def shared_scores(normal, test, fit, cal):
    model = NormalMemory(normal[fit])
    calibration_distances = model.distances(normal[cal])
    test_distances = model.distances(test)
    scores = {
        name: (model.aggregate(calibration), model.aggregate(evaluation))
        for name, calibration, evaluation in zip(
            ("shared_memory", "density_memory"), calibration_distances, test_distances)
    }
    return scores, test_distances[0]


def image_heatmaps(archive, folder, names, labels, distances, score):
    order = np.argsort(score)
    selected = np.r_[order[:3], order[-3:][::-1]]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5), layout="constrained")
    vmin, vmax = np.percentile(distances, [5, 99])
    with ZipFile(archive) as source:
        for ax, i in zip(axes.flat, selected):
            with Image.open(BytesIO(source.read(f"{archive.stem}/{names[i]}"))) as img:
                ax.imshow(img.convert("RGB").resize((224, 224)))
            overlay = ax.imshow(distances[i].reshape(14, 14), cmap="magma",
                                alpha=0.52, extent=(-0.5, 223.5, 223.5, -0.5),
                                interpolation="bilinear", vmin=vmin, vmax=vmax)
            ax.set_title(f"{names[i].split('/')[-1]}  label={labels[i]}  score={score[i]:.3f}")
            ax.axis("off")
    fig.colorbar(overlay, ax=axes, shrink=0.8, label="Mean neighbor distance")
    fig.savefig(folder / "patch_scores.png", dpi=150)
    plt.close(fig)


def image_runs(archive, vision, folder, specialized):
    names = np.array([name.split("/", 1)[1] for name in vision["names"]])
    patches, global_features = vision["patches"], vision["embeddings"]
    results = {}
    for category in ("hazelnut", "zipper"):
        normal = np.flatnonzero(np.char.startswith(names, f"{category}/train/good/"))
        abnormal = np.flatnonzero(np.char.startswith(names, f"{category}/train/bad/"))
        test = np.flatnonzero(np.char.startswith(names, f"{category}/test/"))
        fit, cal = normal_split(len(normal))
        memory, distances = shared_scores(patches[normal], patches[test], fit, cal)
        labels = read_labels(archive, "image_anomaly_labels")
        ids = names[test]
        y = np.array([int(labels[name]["label"] == "bad") for name in ids])
        results[category] = {
            name: calibrated_frame(ids, y, calibration, score)
            for name, (calibration, score) in memory.items()
        }
        if not specialized:
            continue
        calibration, score = memory["shared_memory"]
        target = folder / category
        target.mkdir(exist_ok=True)
        training = np.r_[normal[fit], abnormal]
        train_y = np.isin(training, abnormal).astype(int)
        classifier = SVC(class_weight="balanced").fit(global_features[training], train_y)
        svm_cal = classifier.decision_function(global_features[normal[cal]])
        svm_score = classifier.decision_function(global_features[test])

        def scale(values, reference):
            center = np.median(reference)
            spread = np.quantile(reference, 0.75) - np.quantile(reference, 0.25)
            return (values - center) / max(spread, 1e-8)

        scale_count = 20
        patch_reference, svm_reference = calibration[:scale_count], svm_cal[:scale_count]
        hybrid_cal = (scale(calibration[scale_count:], patch_reference)
                      + scale(svm_cal[scale_count:], svm_reference)) / 2
        hybrid_score = (scale(score, patch_reference) + scale(svm_score, svm_reference)) / 2
        binary_outputs(target, {
            "patch_memory": results[category]["shared_memory"],
            "density_memory": results[category]["density_memory"],
            "global_svm": calibrated_frame(ids, y, svm_cal, svm_score),
            "fusion": calibrated_frame(ids, y, hybrid_cal, hybrid_score),
        }, category)
        image_heatmaps(archive, target, ids, y, distances, score)
        write_json(target / "protocol.json", dict(
            normal_fit=len(fit), normal_calibration=len(cal),
            anomaly_fit=len(abnormal), fusion_scaling=scale_count,
            fusion_calibration=len(hybrid_cal), nominal_fpr=ALPHA))
    return results


def thyroid_run(archive, folder, specialized):
    with ZipFile(archive) as source:
        normal = pd.read_csv(source.open("thyroid/train-set.csv")).to_numpy()
        test_frame = pd.read_csv(source.open("thyroid/test-set.csv"))
    fit, cal = normal_split(len(normal))
    scaler = RobustScaler().fit(normal[fit])
    normal = scaler.transform(normal)
    test = scaler.transform(test_frame.drop(columns="label").to_numpy())
    memory, _ = shared_scores(normal[:, None, :], test[:, None, :], fit, cal)
    y, ids = test_frame["label"].astype(int), test_frame.index + 1
    shared = {
        name: calibrated_frame(ids, y, calibration, score)
        for name, (calibration, score) in memory.items()
    }
    if specialized:
        mixtures = [GaussianMixture(n_components=k, n_init=3, reg_covar=1e-4,
                                    random_state=SEED).fit(normal[fit]) for k in (1, 2, 4, 8)]
        model = min(mixtures, key=lambda mixture: mixture.bic(normal[fit]))
        forest = IsolationForest(n_estimators=300, random_state=SEED).fit(normal[fit])
        binary_outputs(folder, {
            "normal_memory": shared["shared_memory"],
            "density_memory": shared["density_memory"],
            "gaussian_mixture": calibrated_frame(ids, y, -model.score_samples(normal[cal]),
                                                 -model.score_samples(test)),
            "isolation_forest": calibrated_frame(ids, y, -forest.score_samples(normal[cal]),
                                                  -forest.score_samples(test)),
        }, "Thyroid")
        write_json(folder / "protocol.json", dict(
            normal_fit=len(fit), normal_calibration=len(cal), nominal_fpr=ALPHA,
            mixture_bic={str(m.n_components): m.bic(normal[fit]) for m in mixtures},
            mixture_components=model.n_components))
    return shared


def cross_modal(folder, domains):
    for domain, frames in domains.items():
        target = folder / domain
        target.mkdir(exist_ok=True)
        binary_outputs(target, frames, domain)
    pooled = {}
    for name, key in [("pooled_calibrated", "shared_memory"), ("pooled_density", "density_memory")]:
        frame = pd.concat([frames[key].assign(domain=domain) for domain, frames in domains.items()],
                          ignore_index=True)
        pooled[name] = frame.rename(columns={"score": "raw_score"}).assign(
            score=1 - frame["p_normal"])
    binary_outputs(folder, pooled, "Cross-modal anomaly detection")
    summary = pd.concat([
        pd.read_csv(folder / domain / "metrics.csv").assign(domain=domain)
        for domain in domains
    ], ignore_index=True)
    summary.to_csv(folder / "domain_metrics.csv", index=False)
    macro = summary.groupby("model")[["auroc", "average_precision"]].mean()
    write_json(folder / "protocol.json", dict(
        nominal_fpr=ALPHA, normal_calibration_fraction=0.25, neighbors=5,
        density_radius_floor_quantile=0.10,
        macro_auroc=macro.loc["shared_memory", "auroc"],
        macro_average_precision=macro.loc["shared_memory", "average_precision"],
        density_macro_auroc=macro.loc["density_memory", "auroc"],
        density_macro_average_precision=macro.loc["density_memory", "average_precision"]))

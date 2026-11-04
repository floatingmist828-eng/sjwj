import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize

from .common import read_labels


def run(archive, vision, folder):
    names, features = vision["names"], vision["embeddings"]
    ids = [name.rsplit("/", 1)[-1] for name in names]
    clusters = 6
    raw = normalize(features)
    reduced = normalize(PCA(n_components=64, random_state=2026).fit_transform(raw))
    models = {
        "raw_kmeans": (raw, KMeans(clusters, n_init=30, random_state=2026)),
        "pca_kmeans": (reduced, KMeans(clusters, n_init=30, random_state=2026)),
        "pca_ward": (reduced, AgglomerativeClustering(clusters)),
        "pca_spectral": (reduced, SpectralClustering(
            clusters, affinity="rbf", gamma=1.0,
            assign_labels="cluster_qr", random_state=2026)),
    }
    assignments = {name: model.fit_predict(x) for name, (x, model) in models.items()}
    labels = read_labels(archive, "cluster_labels")
    y = np.array([labels[name] for name in ids])
    classes = np.unique(y)
    coords = PCA(n_components=2, random_state=2026).fit_transform(reduced)
    frame = pd.DataFrame(dict(sample=ids, label=y, pc1=coords[:, 0], pc2=coords[:, 1]))
    fig, axes = plt.subplots(1, len(models), figsize=(14, 3.6), layout="constrained")
    metrics = []
    for ax, (name, labels_pred) in zip(axes, assignments.items()):
        table = contingency_matrix(y, labels_pred)
        rows, cols = linear_sum_assignment(table, maximize=True)
        mapping = {col: classes[row] for row, col in zip(rows, cols)}
        metrics.append(dict(model=name, ari=adjusted_rand_score(y, labels_pred),
                            nmi=normalized_mutual_info_score(y, labels_pred),
                            accuracy=table[rows, cols].sum() / len(y),
                            silhouette=silhouette_score(models[name][0], labels_pred)))
        pd.DataFrame(table, index=classes).rename_axis("label").to_csv(
            folder / f"{name}_contingency.csv")
        frame[name] = labels_pred
        matched = np.array([mapping[label] for label in labels_pred])
        frame[f"{name}_matched"] = matched
        for category, marker in zip(classes, ("o", "s", "^", "D", "P", "X")):
            ax.scatter(*coords[matched == category].T, marker=marker,
                       label=category, s=10, alpha=0.75)
        ax.set(title=name, xlabel="PC1", ylabel="PC2")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(classes), loc="outside lower center")
    fig.savefig(folder / "clusters.png", dpi=150)
    plt.close(fig)
    frame.to_csv(folder / "assignments.csv", index=False)
    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)

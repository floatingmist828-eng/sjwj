import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .common import write_json

WINDOW = 12
TARGET = "CO2 (ppm)"


def windows(source):
    raw = pd.read_csv(source, parse_dates=["date"]).sort_values("date", kind="stable")
    data = raw.drop_duplicates("date").set_index("date")
    sentinel_cells = int(data.eq(-9999).to_numpy().sum())
    data = data.replace(-9999, np.nan)
    values = data.to_numpy(dtype=float)
    dates = data.index
    # Require continuity across the history and the forecast timestamp.
    gaps = np.diff(dates.to_numpy()) != np.timedelta64(10, "m")
    valid = ~sliding_window_view(gaps, WINDOW).any(axis=1)
    blocks = sliding_window_view(values, WINDOW + 1, axis=0).transpose(0, 2, 1)
    targets = values[WINDOW:, data.columns.get_loc(TARGET)]
    valid &= np.isfinite(blocks[:, :WINDOW]).all(axis=(1, 2)) & np.isfinite(targets)
    cut_train, cut_val = int(len(data) * 0.70), int(len(data) * 0.85)
    slices = np.searchsorted(dates[[cut_train, cut_val]], dates[WINDOW:][valid], side="right")
    return data, blocks[valid, :WINDOW], targets[valid], dates[WINDOW:][valid], slices, {
        "source_rows": len(raw), "duplicates_removed": len(raw) - len(data),
        "discontinuities": int(gaps.sum()), "invalid_windows": int((~valid).sum()),
        "sentinel_cells": sentinel_cells,
        "train_end": str(dates[cut_train - 1]), "validation_end": str(dates[cut_val - 1]),
        "train_row_cutoff": cut_train,
    }


def summary_features(x):
    return np.concatenate([
        x[:, -1], x[:, -1] - x[:, -2], x[:, -1] - x[:, -4],
        x[:, -1] - x[:, 0], x.mean(axis=1), x.std(axis=1),
        x[:, -3:].mean(axis=1) - x[:, :3].mean(axis=1),
    ], axis=1)


def cyclical_features(history, dates, columns, training_rows, compact):
    wind = np.deg2rad(history[:, :, columns.get_loc("wd (deg)")])
    direction = np.stack([np.sin(wind), np.cos(wind)], axis=-1)
    direction = direction.reshape(len(history), -1)
    direction = StandardScaler().fit(direction[training_rows]).transform(direction)
    phase = 2 * np.pi * (dates.hour.to_numpy() * 60 + dates.minute.to_numpy()) / 1440
    return np.column_stack([compact, direction, np.sin(phase), np.cos(phase)])


def metrics(y, pred):
    return dict(mae=mean_absolute_error(y, pred),
                rmse=np.sqrt(mean_squared_error(y, pred)), r2=r2_score(y, pred),
                bias=np.mean(pred - y))


def run(source, folder):
    data, history, y, dates, partitions, details = windows(source)
    train, validation, test = (partitions == i for i in range(3))
    scale = StandardScaler().fit(data.iloc[:details["train_row_cutoff"]].to_numpy())
    x = scale.transform(history.reshape(-1, history.shape[-1])).reshape(history.shape)
    baseline = history[:, -1, data.columns.get_loc(TARGET)]
    delta = y - baseline
    flat = x.reshape(len(x), -1)
    compact = summary_features(x)
    # Keep raw direction in the baseline; the new model uses its circular encoding.
    wind_column = data.columns.get_loc("wd (deg)")
    other_features = np.arange(compact.shape[1]) % history.shape[-1] != wind_column
    cyclic = cyclical_features(history, dates, data.columns, train, compact[:, other_features])
    ridge = Ridge(alpha=10).fit(flat[train], delta[train])
    predicted = {"persistence": baseline, "ridge_delta": baseline + ridge.predict(flat)}
    for name, features, loss in [
        ("boosted_delta", compact, "squared_error"),
        ("cyclic_boosted_delta", cyclic, "squared_error"),
        ("cyclic_median_delta", cyclic, "absolute_error"),
    ]:
        model = HistGradientBoostingRegressor(
            loss=loss, max_iter=220, learning_rate=0.055, max_leaf_nodes=23,
            min_samples_leaf=25, l2_regularization=2, early_stopping=False,
            random_state=2026)
        model.fit(features[train], delta[train])
        predicted[name] = baseline + model.predict(features)
    predicted["blend_delta"] = (predicted["ridge_delta"] + predicted["boosted_delta"]) / 2
    rows = [
        dict(split=split, model=name, **metrics(y[mask], pred[mask]))
        for split, mask in [("validation", validation), ("test", test)]
        for name, pred in predicted.items()
    ]
    best = min((row for row in rows if row["split"] == "validation"),
               key=lambda row: row["mae"])["model"]
    pd.DataFrame(rows).assign(selected=lambda frame: frame["model"] == best).to_csv(
        folder / "metrics.csv", index=False)
    test_frame = pd.DataFrame(dict(date=dates[test], actual=y[test], last_co2=baseline[test], **{
        name: pred[test] for name, pred in predicted.items()
    }))
    test_frame["actual_change"] = y[test] - baseline[test]
    test_frame["predicted_change"] = predicted[best][test] - baseline[test]
    test_frame["residual"] = test_frame[best] - test_frame["actual"]
    test_frame.to_csv(folder / "test_predictions.csv", index=False)
    window = test_frame[test_frame["date"] < test_frame["date"].iloc[0] + pd.Timedelta(days=2)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")
    axes[0].plot(window["date"], window["actual"], label="Observed")
    axes[0].plot(window["date"], window[best], label=best, alpha=0.8)
    axes[0].set(ylabel="CO2 (ppm)", title="First 48 hours of test forecasts")
    axes[0].legend()
    axes[1].hist(test_frame["residual"], bins=60)
    axes[1].set(xlabel="Prediction minus observation (ppm)", ylabel="Count")
    fig.savefig(folder / "forecast.png", dpi=150)
    plt.close(fig)
    write_json(folder / "protocol.json", dict(
        **details, window=WINDOW, train_windows=int(train.sum()),
        validation_windows=int(validation.sum()), test_windows=int(test.sum()),
        selected=best, selection_metric="validation MAE"))

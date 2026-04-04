from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from power_forecasting_itransformer import (
    build_source_data,
    build_station_frames,
    build_submission_name,
    infer_sample_windows,
    set_seed,
)


def _require_catboost():
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "CatBoost is not installed. Run: pip install catboost"
        ) from exc
    return CatBoostRegressor


LAG_STEPS: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
AGG_WINDOWS: Tuple[int, ...] = (4, 16, 96)


def _metric_nmae(pred: np.ndarray, target: np.ndarray) -> float:
    mae = float(np.mean(np.abs(pred - target)))
    if pred.shape[1] <= 1:
        return mae
    pred_ramp = pred[:, 1:] - pred[:, :-1]
    target_ramp = target[:, 1:] - target[:, :-1]
    ramp_mae = float(np.mean(np.abs(pred_ramp - target_ramp)))
    return 0.8 * mae + 0.2 * ramp_mae


def _build_feature_columns(feature_count: int, history_steps: int) -> List[str]:
    lags = [lag for lag in LAG_STEPS if lag <= history_steps]
    windows = [w for w in AGG_WINDOWS if w <= history_steps]

    cols = ["station_id"]
    cols.extend([f"p_lag_{lag}" for lag in lags])
    cols.extend([f"p_mean_{w}" for w in windows])
    cols.extend([f"p_std_{w}" for w in windows])
    cols.extend([f"p_min_{w}" for w in windows])
    cols.extend([f"p_max_{w}" for w in windows])
    cols.extend([f"p_trend_{w}" for w in windows])

    for i in range(feature_count):
        cols.append(f"f{i}_last")
    for w in windows:
        for i in range(feature_count):
            cols.append(f"f{i}_mean_{w}")
        for i in range(feature_count):
            cols.append(f"f{i}_std_{w}")
    return cols


def _build_row(
    station_id: str,
    feat_hist: np.ndarray,
    p_hist: np.ndarray,
    history_steps: int,
) -> List[object]:
    lags = [lag for lag in LAG_STEPS if lag <= history_steps]
    windows = [w for w in AGG_WINDOWS if w <= history_steps]

    row: List[object] = [station_id]

    row.extend([float(p_hist[-lag]) for lag in lags])
    for w in windows:
        chunk = p_hist[-w:]
        row.append(float(chunk.mean()))
    for w in windows:
        chunk = p_hist[-w:]
        row.append(float(chunk.std()))
    for w in windows:
        chunk = p_hist[-w:]
        row.append(float(chunk.min()))
    for w in windows:
        chunk = p_hist[-w:]
        row.append(float(chunk.max()))
    for w in windows:
        chunk = p_hist[-w:]
        row.append(float(chunk[-1] - chunk[0]))

    row.extend(feat_hist[-1].astype(np.float32).tolist())
    for w in windows:
        chunk = feat_hist[-w:]
        row.extend(chunk.mean(axis=0, dtype=np.float64).astype(np.float32).tolist())
        row.extend(chunk.std(axis=0, dtype=np.float64).astype(np.float32).tolist())

    return row


def _build_tabular_dataset(
    source_data: Dict[str, Dict[str, np.ndarray]],
    history_steps: int,
    horizon_steps: int,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    items: List[Tuple[str, int]] = []
    for station_id, bundle in source_data.items():
        ts = pd.to_datetime(bundle["timestamps"])
        for idx in range(history_steps, len(ts) - horizon_steps + 1):
            current = ts[idx]
            if start_time <= current <= end_time:
                items.append((station_id, idx))

    if not items:
        return pd.DataFrame(columns=_build_feature_columns(len(feature_cols), history_steps)), np.empty((0, horizon_steps), dtype=np.float32)

    rows: List[List[object]] = []
    ys: List[np.ndarray] = []
    for station_id, idx in tqdm(items, desc="build-dataset", leave=False):
        bundle = source_data[station_id]
        feat_hist = bundle["features"][idx - history_steps : idx]
        p_hist = bundle["target"][idx - history_steps : idx]
        y = bundle["target"][idx : idx + horizon_steps]

        if (not np.isfinite(feat_hist).all()) or (not np.isfinite(p_hist).all()) or (not np.isfinite(y).all()):
            continue

        rows.append(_build_row(station_id, feat_hist, p_hist, history_steps))
        ys.append(y.astype(np.float32))

    x = pd.DataFrame(rows, columns=_build_feature_columns(len(feature_cols), history_steps))
    y = np.vstack(ys).astype(np.float32) if ys else np.empty((0, horizon_steps), dtype=np.float32)
    return x, y


@dataclass
class TrainConfig:
    data_root: str = "."
    artifact_dir: str = "artifacts"
    history_steps: int = 96
    train_end: str = "2025-10-31 23:45:00"
    valid_start: str = "2025-11-01 00:00:00"
    valid_end: str = "2025-12-31 23:45:00"
    iterations: int = 1200
    learning_rate: float = 0.03
    depth: int = 8
    l2_leaf_reg: float = 5.0
    random_strength: float = 1.0
    min_data_in_leaf: int = 64
    early_stopping_rounds: int = 120
    used_ram_limit: str | None = None
    thread_count: int = -1
    task_type: str = "CPU"
    devices: str | None = None
    seed: int = 42


def _train_single_model(
    source: str,
    horizon_steps: int,
    source_data: Dict[str, Dict[str, np.ndarray]],
    feature_cols: List[str],
    config: TrainConfig,
    artifact_dir: Path,
) -> None:
    CatBoostRegressor = _require_catboost()

    train_end = pd.Timestamp(config.train_end)
    valid_start = pd.Timestamp(config.valid_start)
    valid_end = pd.Timestamp(config.valid_end)

    x_train, y_train = _build_tabular_dataset(
        source_data=source_data,
        history_steps=config.history_steps,
        horizon_steps=horizon_steps,
        start_time=pd.Timestamp.min,
        end_time=train_end,
        feature_cols=feature_cols,
    )
    x_valid, y_valid = _build_tabular_dataset(
        source_data=source_data,
        history_steps=config.history_steps,
        horizon_steps=horizon_steps,
        start_time=valid_start,
        end_time=valid_end,
        feature_cols=feature_cols,
    )

    if len(x_train) == 0:
        raise ValueError(f"No valid training windows found for source={source}, horizon_steps={horizon_steps}.")
    if len(x_valid) == 0:
        raise ValueError(f"No valid validation windows found for source={source}, horizon_steps={horizon_steps}.")

    model = CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_strength=config.random_strength,
        min_data_in_leaf=config.min_data_in_leaf,
        random_seed=config.seed,
        od_type="Iter",
        od_wait=config.early_stopping_rounds,
        used_ram_limit=config.used_ram_limit,
        thread_count=config.thread_count,
        task_type=config.task_type,
        devices=config.devices,
        verbose=100,
    )

    model.fit(
        x_train,
        y_train,
        cat_features=[0],
        eval_set=(x_valid, y_valid),
        use_best_model=True,
    )

    valid_pred = model.predict(x_valid)
    valid_pred = np.asarray(valid_pred, dtype=np.float32)
    valid_pred = np.clip(valid_pred, 0.0, 1.2)
    valid_score = _metric_nmae(valid_pred, y_valid)
    print(f"[{source}][{horizon_steps}] catboost valid_nmae={valid_score:.6f}")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{source}_{horizon_steps}step.cbm"
    meta_path = artifact_dir / f"{source}_{horizon_steps}step_catboost.json"

    model.save_model(str(model_path))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "catboost",
                "source": source,
                "horizon_steps": horizon_steps,
                "history_steps": config.history_steps,
                "feature_count": len(feature_cols),
                "feature_cols": feature_cols,
                "best_valid_nmae": valid_score,
                "config": asdict(config),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def run_training(args: argparse.Namespace) -> None:
    config = TrainConfig(
        data_root=args.data_root,
        artifact_dir=args.artifact_dir,
        history_steps=args.history_steps,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        min_data_in_leaf=args.min_data_in_leaf,
        early_stopping_rounds=args.early_stopping_rounds,
        used_ram_limit=args.used_ram_limit,
        thread_count=args.thread_count,
        task_type=args.task_type,
        devices=args.devices,
        seed=args.seed,
    )

    set_seed(config.seed)
    station_frames = build_station_frames(Path(config.data_root))
    artifact_dir = Path(config.artifact_dir)
    train_sources = tuple(args.sources)
    train_horizons = tuple(args.horizons)

    for source in train_sources:
        source_data, feature_cols = build_source_data(station_frames, source)
        for horizon_steps in train_horizons:
            gc.collect()
            _train_single_model(
                source=source,
                horizon_steps=horizon_steps,
                source_data=source_data,
                feature_cols=feature_cols,
                config=config,
                artifact_dir=artifact_dir,
            )


def _load_catboost_artifact(artifact_dir: Path, source: str, horizon_steps: int):
    CatBoostRegressor = _require_catboost()

    model_path = artifact_dir / f"{source}_{horizon_steps}step.cbm"
    meta_path = artifact_dir / f"{source}_{horizon_steps}step_catboost.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model, meta


def _predict_for_station(
    bundle: Dict[str, np.ndarray],
    model,
    station_id: str,
    start_time: pd.Timestamp,
    windows: int,
    history_steps: int,
    horizon_steps: int,
) -> np.ndarray:
    ts = pd.to_datetime(bundle["timestamps"])
    full_index = pd.date_range(ts.min(), ts.max(), freq="15min")
    feature_count = bundle["features"].shape[1]

    regularized = pd.DataFrame(
        bundle["features"],
        index=ts,
        columns=[f"feature_{i}" for i in range(feature_count)],
    )
    regularized["target"] = bundle["target"]
    regularized = regularized[~regularized.index.duplicated(keep="last")]
    regularized = regularized.reindex(full_index)

    feature_columns = [f"feature_{i}" for i in range(feature_count)]
    regularized[feature_columns] = regularized[feature_columns].interpolate(limit_direction="both")
    regularized[feature_columns] = regularized[feature_columns].ffill().bfill().fillna(0.0)
    regularized["target"] = regularized["target"].interpolate(limit_direction="both")
    regularized["target"] = regularized["target"].ffill().bfill().fillna(0.0)

    features = regularized[feature_columns].to_numpy(dtype=np.float32)
    target = regularized["target"].to_numpy(dtype=np.float32)
    ts = regularized.index

    ts_to_idx = {timestamp: i for i, timestamp in enumerate(ts)}
    start_indices = []
    current = start_time
    for _ in range(windows):
        if current not in ts_to_idx:
            raise KeyError(f"Timestamp {current} not found for station {station_id}")
        start_indices.append(ts_to_idx[current])
        current += pd.Timedelta(minutes=15)

    rows: List[List[object]] = []
    for idx in tqdm(start_indices, desc=f"predict-{station_id}-{horizon_steps}", leave=False):
        feat_hist = features[idx - history_steps : idx]
        p_hist = target[idx - history_steps : idx]
        if len(feat_hist) < history_steps:
            raise ValueError(f"Not enough history before {ts[idx]} for station {station_id}")
        rows.append(_build_row(station_id, feat_hist, p_hist, history_steps))

    x = pd.DataFrame(rows, columns=_build_feature_columns(feature_count, history_steps))
    pred_norm = np.asarray(model.predict(x), dtype=np.float32)
    pred_norm = np.clip(pred_norm, 0.0, 1.2)

    pred = pred_norm * float(bundle["capacity"])
    pred = np.clip(pred, 0.0, float(bundle["capacity"]))
    return pred[:, :horizon_steps].reshape(-1)


def run_export(args: argparse.Namespace) -> None:
    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    station_frames = build_station_frames(root)
    source_station_frames = {"wind": [], "solar": []}
    for item in station_frames:
        source_station_frames[item.source].append(item)

    artifact_dir = Path(args.artifact_dir)

    for source in ("wind", "solar"):
        source_data, _ = build_source_data(station_frames, source)
        for horizon_steps in (16, 96):
            model, meta = _load_catboost_artifact(artifact_dir, source, horizon_steps)
            history_steps = int(meta["history_steps"])

            for station in source_station_frames[source]:
                file_name = build_submission_name(source, station.met_station_id, horizon_steps)
                template_path = root / args.template_dir / file_name
                if template_path.exists():
                    windows = infer_sample_windows(template_path, horizon_steps)
                elif args.windows is not None:
                    windows = args.windows
                else:
                    raise FileNotFoundError(f"Template not found and --windows not set: {template_path}")

                pred = _predict_for_station(
                    bundle=source_data[station.met_station_id],
                    model=model,
                    station_id=station.met_station_id,
                    start_time=pd.Timestamp(args.start_time),
                    windows=windows,
                    history_steps=history_steps,
                    horizon_steps=horizon_steps,
                )

                df = pd.DataFrame(
                    {
                        "id": np.arange(len(pred), dtype=np.int64),
                        "pred": pred.astype(np.float32),
                        "type": np.full(len(pred), args.submission_type, dtype=np.int64),
                    }
                )
                df.to_csv(output_dir / file_name, index=False, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Power forecasting training and submission export (CatBoost).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train wind and solar CatBoost multi-horizon models.")
    train_parser.add_argument("--data-root", default=".")
    train_parser.add_argument("--artifact-dir", default="artifacts")
    train_parser.add_argument("--history-steps", type=int, default=96)
    train_parser.add_argument("--train-end", default="2025-10-31 23:45:00")
    train_parser.add_argument("--valid-start", default="2025-11-01 00:00:00")
    train_parser.add_argument("--valid-end", default="2025-12-31 23:45:00")
    train_parser.add_argument("--iterations", type=int, default=1200)
    train_parser.add_argument("--learning-rate", type=float, default=0.03)
    train_parser.add_argument("--depth", type=int, default=8)
    train_parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    train_parser.add_argument("--random-strength", type=float, default=1.0)
    train_parser.add_argument("--min-data-in-leaf", type=int, default=64)
    train_parser.add_argument("--early-stopping-rounds", type=int, default=120)
    train_parser.add_argument("--used-ram-limit", type=str, default=None)
    train_parser.add_argument("--thread-count", type=int, default=-1)
    train_parser.add_argument("--task-type", type=str, choices=("CPU", "GPU"), default="CPU")
    train_parser.add_argument("--devices", type=str, default=None)
    train_parser.add_argument("--sources", nargs="+", choices=("wind", "solar"), default=("wind", "solar"))
    train_parser.add_argument("--horizons", nargs="+", type=int, choices=(16, 96), default=(16, 96))
    train_parser.add_argument("--seed", type=int, default=42)

    export_parser = subparsers.add_parser("export", help="Export 20 submission csv files using CatBoost artifacts.")
    export_parser.add_argument("--data-root", default=".")
    export_parser.add_argument("--artifact-dir", default="artifacts")
    export_parser.add_argument("--output-dir", default="submission")
    export_parser.add_argument("--template-dir", default="a_pred")
    export_parser.add_argument("--start-time", default="2025-11-01 00:00:00")
    export_parser.add_argument("--windows", type=int, default=None)
    export_parser.add_argument("--submission-type", type=int, default=0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_training(args)
    elif args.command == "export":
        run_export(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

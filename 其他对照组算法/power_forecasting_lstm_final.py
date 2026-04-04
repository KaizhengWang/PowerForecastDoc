# ==============================
# 使用方法
# ==============================

# 训练模型（默认当前目录为数据目录）
# 会自动识别CSV → 训练风电/光伏4个模型 → 保存到 artifacts/
# 命令：
# python power_forecasting_lstm_final.py train

# 可选参数示例：
# .\.venv\Scripts\python.exe .\power_forecasting_lstm_final.py train --epochs 12 --batch-size 1024 --early-stopping-patience 4

# --------------------------------

# 生成预测结果（提交文件）
# 会加载 artifacts/ 下模型 → 预测 → 输出到 submission/
# 命令：
# python power_forecasting_lstm_final.py export

# 可选参数示例：
# python power_forecasting_lstm_final.py export --data-root ./data --output-dir ./submission --start-time "2025-11-01 00:00:00"

# --------------------------------

# 标准流程
# 先训练，再预测：
# python power_forecasting_lstm_final.py train
# python power_forecasting_lstm_final.py export

# --------------------------------

#   注意：
# - 所有CSV需与此脚本放在同一目录
# - 无GPU会自动使用CPU（速度较慢）

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


TIME_COL = "时间"
POWER_STATION_COL = "厂站id"
POWER_COL = "功率"
WIND_MET_COL = "测风塔id"
SOLAR_MET_COL = "光伏气象站ID"
CAPACITY_COL = "装机"
WIND_GAP_FILL_BLEND_WEIGHT = 0.12
DEFAULT_ERA5_DROPOUT_PROB = 0.3
WIND_ERA_TARGET_COLUMNS = [
    "10米风速",
    "10米风向",
    "30米风速",
    "30米风向",
    "50米风速",
    "50米风向",
    "70米风速",
    "70米风向",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_columns(columns: Iterable[str]) -> List[str]:
    return [str(col).replace("\ufeff", "").replace("\n", "").replace("\r", "").strip() for col in columns]


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.columns = clean_columns(df.columns)
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def _normalize_lon_lat(value: object) -> str:
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-6:
        return str(int(round(numeric)))
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def _wind_direction_from_uv(u: pd.Series, v: pd.Series) -> pd.Series:
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return pd.Series(direction, index=u.index, dtype=np.float32)


def _circular_blend_deg(base_deg: pd.Series, ref_deg: pd.Series, ref_weight: float) -> pd.Series:
    base_rad = np.deg2rad(base_deg.to_numpy(dtype=np.float64))
    ref_rad = np.deg2rad(ref_deg.to_numpy(dtype=np.float64))
    x = (1.0 - ref_weight) * np.cos(base_rad) + ref_weight * np.cos(ref_rad)
    y = (1.0 - ref_weight) * np.sin(base_rad) + ref_weight * np.sin(ref_rad)
    blended = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return pd.Series(blended.astype(np.float32), index=base_deg.index)


def load_era5_wind_reference(root: Path) -> Dict[str, pd.DataFrame]:
    candidate_roots = [root / "era5csv"]
    if root.parent != root:
        candidate_roots.append(root.parent / "era5csv")
    era_root = next((candidate for candidate in candidate_roots if candidate.exists()), None)
    if era_root is None:
        return {}

    references: Dict[str, pd.DataFrame] = {}
    exponent = 0.14
    for path in sorted(era_root.glob("*.csv")):
        df = read_csv_auto(path)
        df.columns = clean_columns(df.columns)
        if not {"valid_time", "u100", "v100"}.issubset(df.columns):
            continue

        df["valid_time"] = pd.to_datetime(df["valid_time"]) + pd.Timedelta(hours=8)
        speed100 = np.sqrt(pd.to_numeric(df["u100"], errors="coerce") ** 2 + pd.to_numeric(df["v100"], errors="coerce") ** 2)
        direction100 = _wind_direction_from_uv(
            pd.to_numeric(df["u100"], errors="coerce"),
            pd.to_numeric(df["v100"], errors="coerce"),
        )

        era_frame = pd.DataFrame({TIME_COL: df["valid_time"]})
        for height in (10, 30, 50, 70):
            scaled_speed = speed100 * np.power(height / 100.0, exponent)
            era_frame[f"{height}米风速"] = scaled_speed.astype(np.float32)
            era_frame[f"{height}米风向"] = direction100.astype(np.float32)

        latitude = _normalize_lon_lat(df["latitude"].iloc[0])
        longitude = _normalize_lon_lat(df["longitude"].iloc[0])
        era_frame = era_frame.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="last")
        references[f"{longitude}_{latitude}"] = era_frame

    return references


def blend_wind_features_with_era5(
    df: pd.DataFrame,
    era_reference: pd.DataFrame | None,
    blend_weight: float = WIND_GAP_FILL_BLEND_WEIGHT,
) -> pd.DataFrame:
    if era_reference is None or era_reference.empty:
        return df

    blended = df.copy()
    merged_ref = era_reference.copy().set_index(TIME_COL).resample("15min").interpolate("time").ffill().bfill().reset_index()
    merged = blended.merge(merged_ref, on=TIME_COL, how="left", suffixes=("", "__era5"))

    for column in WIND_ERA_TARGET_COLUMNS:
        if column not in merged.columns:
            continue
        original = pd.to_numeric(merged[column], errors="coerce")
        missing_mask = original.isna()
        if not missing_mask.any():
            continue

        context_fill = original.interpolate(limit_direction="both").ffill().bfill()
        era_column = f"{column}__era5"
        era_values = pd.to_numeric(merged[era_column], errors="coerce") if era_column in merged.columns else None
        if era_values is None:
            merged[column] = context_fill
            continue

        if "风向" in column:
            blend_mask = missing_mask & era_values.notna() & context_fill.notna()
            context_for_angle = context_fill.where(context_fill.notna(), era_values)
            merged[column] = context_fill
            if blend_mask.any():
                merged.loc[blend_mask, column] = _circular_blend_deg(
                    context_for_angle.loc[blend_mask],
                    era_values.loc[blend_mask],
                    blend_weight,
                )
            fallback_mask = missing_mask & merged[column].isna() & era_values.notna()
            merged.loc[fallback_mask, column] = era_values.loc[fallback_mask]
        else:
            merged[column] = context_fill
            blend_mask = missing_mask & era_values.notna() & context_fill.notna()
            merged.loc[blend_mask, column] = (
                (1.0 - blend_weight) * context_fill.loc[blend_mask] + blend_weight * era_values.loc[blend_mask]
            )
            fallback_mask = missing_mask & merged[column].isna() & era_values.notna()
            merged.loc[fallback_mask, column] = era_values.loc[fallback_mask]

    drop_columns = [f"{column}__era5" for column in WIND_ERA_TARGET_COLUMNS if f"{column}__era5" in merged.columns]
    return merged.drop(columns=drop_columns)


@dataclass
class DataFiles:
    power: Path
    wind_features: Path
    solar_features: Path
    wind_mapping: Path
    solar_mapping: Path


def detect_data_files(root: Path) -> DataFiles:
    roles: Dict[str, Path] = {}
    all_csvs = list(root.glob("*.csv"))

    for path in all_csvs:
        try:
            # 读取表头，尝试多种编码
            df_head = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df_head = pd.read_csv(path, nrows=0, encoding="gb18030")

        # 归一化处理：去掉空格、BOM，并全部转为小写
        header = [str(c).strip().replace('\ufeff', '').lower() for c in df_head.columns]
        header_set = set(header)

        # 准备小写化的常量进行匹配
        l_time = TIME_COL.lower()
        l_station = POWER_STATION_COL.lower()
        l_power = POWER_COL.lower()
        l_wind_met = WIND_MET_COL.lower()
        l_solar_met = SOLAR_MET_COL.lower()

        # --- 识别逻辑 ---

        # 1. 功率数据：包含时间、厂站id、功率值
        if {l_time, l_station, l_power}.issubset(header_set):
            roles["power"] = path

        # 2. 映射表识别：包含 测风塔id 和 厂站id (且不含时间列，防止误判为特征表)
        elif l_wind_met in header_set and l_station in header_set and l_time not in header_set:
            roles["wind_mapping"] = path

        # 3. 光伏映射表：优先于光伏特征识别，避免“光伏气象站-场站.csv”被文件名规则误判
        elif ("气象站id" in header_set or l_solar_met in header_set) and l_station in header_set and l_time not in header_set:
            roles["solar_mapping"] = path

        # 4. 风电特征：包含 测风塔id 和 时间
        elif l_time in header_set and l_wind_met in header_set:
            roles["wind_features"] = path

        # 5. 光伏特征：包含 光伏气象站ID 和 时间
        elif l_time in header_set and l_solar_met in header_set:
            roles["solar_features"] = path

        # 6. 文件名兜底
        elif "测风塔" in path.name and "场站" in path.name:
            roles["wind_mapping"] = path
        elif "测风塔" in path.name:
            roles["wind_features"] = path
        elif "光伏气象站" in path.name and "场站" in path.name:
            roles["solar_mapping"] = path
        elif "测光塔" in path.name or "光伏气象站" in path.name:
            roles["solar_features"] = path

    # 调试输出：如果报错，让你知道它到底抓到了哪些文件
    print(f"--- 自动识别结果 ---")
    for role, p in roles.items():
        print(f"角色 [{role}] -> 文件: {p.name}")

    # DataFiles 需要五类文件，这里必须和 dataclass 保持一致
    required = ["power", "wind_features", "solar_features", "wind_mapping", "solar_mapping"]
    missing = [name for name in required if name not in roles]

    if missing:
        print(f"当前目录下所有CSV: {[f.name for f in all_csvs]}")
        raise FileNotFoundError(f"无法识别必要的数据文件角色: {missing}。请检查文件表头或文件名。")

    return DataFiles(**roles)

@dataclass
class StationFrame:
    source: str
    met_station_id: str
    plant_id: str
    capacity: float
    frame: pd.DataFrame


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    minutes = df[TIME_COL].dt.hour * 60 + df[TIME_COL].dt.minute
    quarter = minutes / (24 * 60)
    day_of_year = (df[TIME_COL].dt.dayofyear - 1) / 366.0
    df["tod_sin"] = np.sin(2 * np.pi * quarter)
    df["tod_cos"] = np.cos(2 * np.pi * quarter)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year)
    df["month"] = df[TIME_COL].dt.month.astype(np.float32)
    df["hour"] = df[TIME_COL].dt.hour.astype(np.float32)
    return df


def add_domain_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    def add_wind_dir_components(column: str, prefix: str) -> None:
        if column in df.columns:
            radians = np.deg2rad(pd.to_numeric(df[column], errors="coerce"))
            df[f"{prefix}_sin"] = np.sin(radians)
            df[f"{prefix}_cos"] = np.cos(radians)

    if source == "wind":
        speed_cols = [col for col in ["10米风速", "30米风速", "50米风速", "70米风速", "轮毂高度风速"] if col in df.columns]
        for col in ["10米风向", "30米风向", "50米风向", "70米风向", "轮毂高度风向"]:
            add_wind_dir_components(col, col.replace("风向", "风向分量"))
        if speed_cols:
            numeric_speeds = df[speed_cols].apply(pd.to_numeric, errors="coerce")
            df["风速均值"] = numeric_speeds.mean(axis=1)
            df["风速标准差"] = numeric_speeds.std(axis=1).fillna(0.0)
        if {"10米风速", "轮毂高度风速"}.issubset(df.columns):
            low_speed = pd.to_numeric(df["10米风速"], errors="coerce").clip(lower=0.1)
            hub_speed = pd.to_numeric(df["轮毂高度风速"], errors="coerce")
            df["轮毂10米风速比"] = hub_speed / low_speed
        if {"10m气压", "10m气温", "轮毂高度风速"}.issubset(df.columns):
            pressure_pa = pd.to_numeric(df["10m气压"], errors="coerce") * 100.0
            temp_k = pd.to_numeric(df["10m气温"], errors="coerce") + 273.15
            air_density = pressure_pa / (287.05 * temp_k.clip(lower=200.0))
            hub_speed = pd.to_numeric(df["轮毂高度风速"], errors="coerce").clip(lower=0.0)
            df["空气密度估计"] = air_density
            df["风功率密度估计"] = 0.5 * air_density * np.power(hub_speed, 3)
    elif source == "solar":
        add_wind_dir_components("风向", "风向分量")
        if {"总辐照度", "直接辐照度"}.issubset(df.columns):
            ghi = pd.to_numeric(df["总辐照度"], errors="coerce").clip(lower=1.0)
            dni = pd.to_numeric(df["直接辐照度"], errors="coerce")
            df["直接辐照占比"] = dni / ghi
        if {"总辐照度", "间接辐照度"}.issubset(df.columns):
            ghi = pd.to_numeric(df["总辐照度"], errors="coerce").clip(lower=1.0)
            dhi = pd.to_numeric(df["间接辐照度"], errors="coerce")
            df["散射辐照占比"] = dhi / ghi
        if {"光伏电池板温度", "温度"}.issubset(df.columns):
            panel_temp = pd.to_numeric(df["光伏电池板温度"], errors="coerce")
            ambient_temp = pd.to_numeric(df["温度"], errors="coerce")
            df["板温环境温差"] = panel_temp - ambient_temp
        if "总辐照度" in df.columns:
            ghi = pd.to_numeric(df["总辐照度"], errors="coerce")
            df["白天指示"] = (ghi > 5.0).astype(np.float32)
            df["辐照平方根"] = np.sqrt(ghi.clip(lower=0.0))
    return df


def clean_numeric_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
    return df


def sanitize_target(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    series = series.interpolate(limit_direction="both")
    series = series.ffill().bfill().fillna(0.0)
    return series


def build_station_frames(
    root: Path,
    era5_dropout_prob: float = 0.0,
    training_mode: bool = False,
) -> List[StationFrame]:
    files = detect_data_files(root)
    era5_references = load_era5_wind_reference(root)

    # 辅助函数：统一清洗列名并转为小写，确保内存中逻辑一致
    def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.lower()
        return df

    # 1. 读取并立即归一化列名
    power_df = _prep_df(read_csv_auto(files.power))
    wind_df = _prep_df(read_csv_auto(files.wind_features))
    solar_df = _prep_df(read_csv_auto(files.solar_features))
    wind_map = _prep_df(read_csv_auto(files.wind_mapping))
    solar_map = _prep_df(read_csv_auto(files.solar_mapping))

    # 2. 统一使用小写化的常量名进行后续逻辑
    # 假设原常量 WIND_MET_COL = "测风塔ID"，转为小写后可匹配所有文件
    l_wind_met = WIND_MET_COL.lower()  # '测风塔id'
    l_solar_met = SOLAR_MET_COL.lower()  # '气象站id' (或对应的常量)
    l_plant_col = POWER_STATION_COL.lower()  # '厂站id'
    l_time_col = TIME_COL.lower()  # 'time'
    l_cap_col = CAPACITY_COL.lower()  # '装机'
    l_pwr_col = POWER_COL.lower()  # 'power'

    # 3. 容错性检查（使用小写匹配）
    def check_col(df, col_name, file_tag):
        if col_name not in df.columns:
            raise KeyError(f"在 {file_tag} 中找不到列名 '{col_name}'。实际列名: {df.columns.tolist()}")

    check_col(wind_map, l_wind_met, "wind_mapping")
    check_col(wind_map, l_plant_col, "wind_mapping")
    check_col(solar_map, l_plant_col, "solar_mapping")

    # 4. 数据类型转换
    power_df[l_time_col] = pd.to_datetime(power_df[l_time_col])
    wind_df[l_time_col] = pd.to_datetime(wind_df[l_time_col])
    solar_df[l_time_col] = pd.to_datetime(solar_df[l_time_col])
    actual_solar_key = l_solar_met if l_solar_met in solar_map.columns else "气象站id"

    for df in [power_df, wind_df, solar_df, wind_map, solar_map]:
        # 将所有 ID 类列强转为 string，防止 1001 被识别为 int
        for c in [l_wind_met, l_solar_met, l_plant_col, actual_solar_key]:
            if c in df.columns:
                df[c] = df[c].astype(str)

    # 5. 组装 StationFrame
    station_frames: List[StationFrame] = []

    # 处理风电
    for _, row in wind_map.iterrows():
        met_id, plant_id = row[l_wind_met], row[l_plant_col]
        capacity = float(row[l_cap_col])
        era5_key = None
        if "气象站经度" in row.index and "气象站纬度" in row.index:
            era5_key = f"{_normalize_lon_lat(row['气象站经度'])}_{_normalize_lon_lat(row['气象站纬度'])}"

        feat = wind_df[wind_df[l_wind_met] == met_id].copy()
        pwr = power_df[power_df[l_plant_col] == plant_id].copy()

        if feat.empty or pwr.empty: continue

        merged = feat.merge(pwr[[l_time_col, l_pwr_col]], on=l_time_col, how="inner").sort_values(l_time_col)
        use_era5 = True
        if training_mode and era5_dropout_prob > 0.0 and random.random() < era5_dropout_prob:
            use_era5 = False
        merged = blend_wind_features_with_era5(merged, era5_references.get(era5_key) if use_era5 else None)
        merged["capacity"] = capacity
        merged["power_norm"] = merged[l_pwr_col] / max(capacity, 1e-6)
        merged = add_time_features(merged)
        merged = add_domain_features(merged, source="wind")

        station_frames.append(StationFrame(
            source="wind", met_station_id=met_id, plant_id=plant_id, capacity=capacity, frame=merged
        ))

    # 处理光伏
    # 兼容处理：检查 solar_map 里是用 '气象站id' 还是常量定义的值
    for _, row in solar_map.iterrows():
        met_id, plant_id = row[actual_solar_key], row[l_plant_col]
        capacity = float(row[l_cap_col])

        feat = solar_df[solar_df[l_solar_met] == met_id].copy()
        pwr = power_df[power_df[l_plant_col] == plant_id].copy()

        if feat.empty or pwr.empty: continue

        merged = feat.merge(pwr[[l_time_col, l_pwr_col]], on=l_time_col, how="inner").sort_values(l_time_col)
        merged["capacity"] = capacity
        merged["power_norm"] = merged[l_pwr_col] / max(capacity, 1e-6)
        merged = add_time_features(merged)
        merged = add_domain_features(merged, source="solar")

        station_frames.append(StationFrame(
            source="solar", met_station_id=met_id, plant_id=plant_id, capacity=capacity, frame=merged
        ))

    return station_frames


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        TIME_COL,
        POWER_COL,
        "power_norm",
        "capacity",
        WIND_MET_COL,
        SOLAR_MET_COL,
        POWER_STATION_COL,
    }
    numeric_cols = [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]
    return numeric_cols


def build_source_data(station_frames: List[StationFrame], source: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    source_frames = [station for station in station_frames if station.source == source]
    feature_cols = select_feature_columns(source_frames[0].frame)
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for station in source_frames:
        df = station.frame.copy()
        df = clean_numeric_frame(df, feature_cols + ["power_norm"])
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)
        df["power_norm"] = sanitize_target(df["power_norm"]).clip(0.0, 1.5)
        features = df[feature_cols].to_numpy(dtype=np.float32)
        target = df["power_norm"].to_numpy(dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features remain after cleaning for station {station.met_station_id}")
        if not np.isfinite(target).all():
            raise ValueError(f"Non-finite targets remain after cleaning for station {station.met_station_id}")
        timestamps = df[TIME_COL].to_numpy(dtype="datetime64[ns]")
        data[station.met_station_id] = {
            "features": features,
            "target": target,
            "timestamps": timestamps,
            "capacity": np.float32(station.capacity),
        }
    return data, feature_cols


def fit_feature_scaler(source_data: Dict[str, Dict[str, np.ndarray]], cutoff_time: pd.Timestamp) -> Dict[str, np.ndarray]:
    chunks = []
    for bundle in source_data.values():
        ts = pd.to_datetime(bundle["timestamps"])
        mask = ts <= cutoff_time
        train_chunk = bundle["features"][mask]
        if len(train_chunk):
            chunks.append(train_chunk)
    if not chunks:
        raise ValueError("No training samples available before scaler cutoff time.")
    stacked = np.concatenate(chunks, axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError("Non-finite values detected while fitting feature scaler.")
    mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = stacked.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


class WindowDataset(Dataset):
    def __init__(
        self,
        source_data: Dict[str, Dict[str, np.ndarray]],
        station_to_idx: Dict[str, int],
        history_steps: int,
        horizon_steps: int,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        scaler: Dict[str, np.ndarray],
    ) -> None:
        self.items: List[Tuple[str, int]] = []
        self.source_data = source_data
        self.station_to_idx = station_to_idx
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.scaler = scaler

        for station_id, bundle in source_data.items():
            ts = pd.to_datetime(bundle["timestamps"])
            for idx in range(history_steps, len(ts) - horizon_steps + 1):
                current = ts[idx]
                if start_time <= current <= end_time:
                    feature_slice = bundle["features"][idx - history_steps : idx]
                    target_slice = bundle["target"][idx : idx + horizon_steps]
                    if not np.isfinite(feature_slice).all() or not np.isfinite(target_slice).all():
                        continue
                    self.items.append((station_id, idx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        station_id, idx = self.items[index]
        bundle = self.source_data[station_id]
        feature_slice = bundle["features"][idx - self.history_steps : idx]
        scaled_features = (feature_slice - self.scaler["mean"]) / self.scaler["std"]
        power_history = bundle["target"][idx - self.history_steps : idx, None]
        x = np.concatenate([scaled_features, power_history], axis=1)
        y = bundle["target"][idx : idx + self.horizon_steps]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Encountered non-finite sample for station {station_id} at index {idx}")
        station_idx = self.station_to_idx[station_id]
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.tensor(station_idx, dtype=torch.long),
            torch.from_numpy(y.astype(np.float32)),
        )


class LegacyGRUModel(nn.Module):
    def __init__(self, input_dim: int, history_steps: int, horizon_steps: int, station_count: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.station_embedding = nn.Embedding(station_count, 16)
        self.history_projection = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 16 + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, horizon_steps),
        )

    def forward(self, x: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        summary = encoded[:, -1, :]
        station_vec = self.station_embedding(station_idx)
        power_history = x[:, :, -1]
        recent_1h = power_history[:, -4:].mean(dim=1, keepdim=True)
        recent_4h = power_history[:, -16:].mean(dim=1, keepdim=True)
        recent_24h = power_history.mean(dim=1, keepdim=True)
        latest_power = power_history[:, -1:].contiguous()
        history_stats = self.history_projection(torch.cat([latest_power, recent_1h, recent_4h, recent_24h], dim=1))
        return self.head(torch.cat([summary, station_vec, history_stats], dim=1))


class LSTMForecastModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        history_steps: int,
        horizon_steps: int,
        station_count: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.station_embedding = nn.Embedding(station_count, 16)
        self.history_projection = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 16 + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_steps),
        )

    def forward(self, x: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        summary = encoded[:, -1, :]
        station_vec = self.station_embedding(station_idx)
        power_history = x[:, :, -1]
        recent_1h = power_history[:, -4:].mean(dim=1, keepdim=True)
        recent_4h = power_history[:, -16:].mean(dim=1, keepdim=True)
        recent_24h = power_history.mean(dim=1, keepdim=True)
        latest_power = power_history[:, -1:].contiguous()
        history_stats = self.history_projection(torch.cat([latest_power, recent_1h, recent_4h, recent_24h], dim=1))
        return self.head(torch.cat([summary, station_vec, history_stats], dim=1))

DirectMultiHorizonModel = LSTMForecastModel


@dataclass
class TrainConfig:
    data_root: str = "."
    artifact_dir: str = "artifacts"
    history_steps: int = 96
    train_end: str = "2025-10-31 23:45:00"
    valid_start: str = "2025-11-01 00:00:00"
    valid_end: str = "2025-12-31 23:45:00"
    batch_size: int = 1024
    epochs: int = 6
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    early_stopping_patience: int = 0
    seed: int = 42
    num_workers: int = 2
    era5_dropout_prob: float = DEFAULT_ERA5_DROPOUT_PROB


def nmae_on_normalized(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mae = torch.mean(torch.abs(pred - target))
    if pred.shape[1] <= 1:
        return mae
    pred_ramp = pred[:, 1:] - pred[:, :-1]
    target_ramp = target[:, 1:] - target[:, :-1]
    ramp_mae = torch.mean(torch.abs(pred_ramp - target_ramp))
    return 0.8 * mae + 0.2 * ramp_mae


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for x, station_idx, y in loader:
            x = x.to(device)
            station_idx = station_idx.to(device)
            y = y.to(device)
            pred = model(x, station_idx).clamp_(0.0, 1.2)
            losses.append(float(nmae_on_normalized(pred, y).detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


def train_single_model(
    source: str,
    horizon_steps: int,
    source_data: Dict[str, Dict[str, np.ndarray]],
    feature_cols: List[str],
    config: TrainConfig,
    device: torch.device,
    artifact_dir: Path,
) -> None:
    train_end = pd.Timestamp(config.train_end)
    valid_start = pd.Timestamp(config.valid_start)
    valid_end = pd.Timestamp(config.valid_end)
    scaler = fit_feature_scaler(source_data, train_end)
    station_ids = sorted(source_data.keys())
    station_to_idx = {station_id: i for i, station_id in enumerate(station_ids)}

    train_ds = WindowDataset(
        source_data=source_data,
        station_to_idx=station_to_idx,
        history_steps=config.history_steps,
        horizon_steps=horizon_steps,
        start_time=pd.Timestamp.min,
        end_time=train_end,
        scaler=scaler,
    )
    valid_ds = WindowDataset(
        source_data=source_data,
        station_to_idx=station_to_idx,
        history_steps=config.history_steps,
        horizon_steps=horizon_steps,
        start_time=valid_start,
        end_time=valid_end,
        scaler=scaler,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    if len(train_ds) == 0:
        raise ValueError(f"No valid training windows found for source={source}, horizon_steps={horizon_steps}.")
    if len(valid_ds) == 0:
        raise ValueError(f"No valid validation windows found for source={source}, horizon_steps={horizon_steps}.")

    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DirectMultiHorizonModel(
        input_dim=len(feature_cols) + 1,
        history_steps=config.history_steps,
        horizon_steps=horizon_steps,
        station_count=len(station_ids),
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = float("inf")
    best_state = None
    no_improve_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"{source}-{horizon_steps} epoch {epoch}", leave=False)
        for x, station_idx, y in progress:
            x = x.to(device, non_blocking=True)
            station_idx = station_idx.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(x, station_idx)
                loss = nmae_on_normalized(pred, y)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected for source={source}, horizon_steps={horizon_steps}")
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.5f}")

        valid_score = evaluate_model(model, valid_loader, device)
        print(f"[{source}][{horizon_steps}] epoch={epoch} valid_nmae={valid_score:.6f}")
        if valid_score < best_score:
            best_score = valid_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if config.early_stopping_patience > 0 and no_improve_epochs >= config.early_stopping_patience:
                print(
                    f"[{source}][{horizon_steps}] early stopping at epoch={epoch}, "
                    f"best_valid_nmae={best_score:.6f}"
                )
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": best_state,
        "model_type": "lstm",
        "source": source,
        "horizon_steps": horizon_steps,
        "history_steps": config.history_steps,
        "feature_cols": feature_cols,
        "station_ids": station_ids,
        "station_to_idx": station_to_idx,
        "scaler_mean": scaler["mean"],
        "scaler_std": scaler["std"],
        "config": asdict(config),
        "best_valid_nmae": best_score,
        "hidden_dim": config.hidden_dim,
    }
    target_path = artifact_dir / f"{source}_{horizon_steps}step.pt"
    torch.save(payload, target_path)
    with (artifact_dir / f"{source}_{horizon_steps}step.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": source,
                "horizon_steps": horizon_steps,
                "history_steps": config.history_steps,
                "feature_count": len(feature_cols),
                "station_count": len(station_ids),
                "best_valid_nmae": best_score,
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
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        num_workers=args.num_workers,
        era5_dropout_prob=args.era5_dropout_prob,
    )
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    station_frames = build_station_frames(
        Path(config.data_root),
        era5_dropout_prob=config.era5_dropout_prob,
        training_mode=True,
    )
    artifact_dir = Path(config.artifact_dir)

    for source in ("wind", "solar"):
        source_data, feature_cols = build_source_data(station_frames, source)
        for horizon_steps in (16, 96):
            train_single_model(
                source=source,
                horizon_steps=horizon_steps,
                source_data=source_data,
                feature_cols=feature_cols,
                config=config,
                device=device,
                artifact_dir=artifact_dir,
            )


def load_artifact(path: Path, device: torch.device) -> Dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_type = payload.get("model_type", "lstm")
    if model_type == "lstm":
        model_cls = LSTMForecastModel
    elif model_type == "itransformer":
        raise ValueError(
            f"Artifact {path} is itransformer. "
            "Please use power_forecasting_itransformer_final.py export, "
            "or retrain with power_forecasting_lstm_final.py train."
        )
    else:
        model_cls = LegacyGRUModel
    model = model_cls(
        input_dim=len(payload["feature_cols"]) + 1,
        history_steps=int(payload["history_steps"]),
        horizon_steps=int(payload["horizon_steps"]),
        station_count=len(payload["station_ids"]),
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    payload["model"] = model
    return payload


def infer_sample_windows(template_path: Path, horizon_steps: int) -> int:
    with template_path.open("r", encoding="utf-8-sig", newline="") as f:
        row_count = sum(1 for _ in f) - 1
    return row_count // horizon_steps


def predict_for_station(
    bundle: Dict[str, np.ndarray],
    artifact: Dict[str, object],
    station_id: str,
    start_time: pd.Timestamp,
    windows: int,
    device: torch.device,
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
    ts = regularized.index
    features = regularized[feature_columns].to_numpy(dtype=np.float32)
    target = regularized["target"].to_numpy(dtype=np.float32)

    start_indices = []
    current = start_time
    ts_to_idx = {timestamp: i for i, timestamp in enumerate(ts)}
    for _ in range(windows):
        if current not in ts_to_idx:
            raise KeyError(f"Timestamp {current} not found for station {station_id}")
        start_indices.append(ts_to_idx[current])
        current += pd.Timedelta(minutes=15)

    history_steps = int(artifact["history_steps"])
    scaler_mean = artifact["scaler_mean"]
    scaler_std = artifact["scaler_std"]
    station_to_idx = artifact["station_to_idx"]
    model = artifact["model"]
    horizon_steps = int(artifact["horizon_steps"])

    outputs = []
    with torch.no_grad():
        for idx in tqdm(start_indices, desc=f"predict-{station_id}-{horizon_steps}", leave=False):
            feature_slice = features[idx - history_steps : idx]
            if len(feature_slice) < history_steps:
                raise ValueError(f"Not enough history before {ts[idx]} for station {station_id}")
            scaled_features = (feature_slice - scaler_mean) / scaler_std
            power_history = target[idx - history_steps : idx, None]
            x = np.concatenate([scaled_features, power_history], axis=1)
            x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
            station_tensor = torch.tensor([station_to_idx[station_id]], dtype=torch.long, device=device)
            pred_norm = model(x_tensor, station_tensor).squeeze(0).detach().cpu().numpy()
            pred_norm = np.clip(pred_norm, 0.0, 1.2)
            pred = pred_norm * float(bundle["capacity"])
            pred = np.clip(pred, 0.0, float(bundle["capacity"]))
            outputs.append(pred[:horizon_steps])
    return np.concatenate(outputs, axis=0)


def build_submission_name(source: str, station_id: str, horizon_steps: int) -> str:
    prefix = "风电" if source == "wind" else "光电"
    suffix = "4h" if horizon_steps == 16 else "24h"
    return f"{prefix}_{station_id}_{suffix}.csv"


def run_export(args: argparse.Namespace) -> None:
    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    station_frames = build_station_frames(root, era5_dropout_prob=0.0, training_mode=False)
    source_station_frames = {"wind": [], "solar": []}
    for item in station_frames:
        source_station_frames[item.source].append(item)

    for source in ("wind", "solar"):
        source_data, _ = build_source_data(station_frames, source)
        for horizon_steps in (16, 96):
            artifact = load_artifact(Path(args.artifact_dir) / f"{source}_{horizon_steps}step.pt", device)
            for station in source_station_frames[source]:
                file_name = build_submission_name(source, station.met_station_id, horizon_steps)
                template_path = root / args.template_dir / file_name
                if template_path.exists():
                    windows = infer_sample_windows(template_path, horizon_steps)
                elif args.windows is not None:
                    windows = args.windows
                else:
                    raise FileNotFoundError(f"Template not found and --windows not set: {template_path}")

                pred = predict_for_station(
                    bundle=source_data[station.met_station_id],
                    artifact=artifact,
                    station_id=station.met_station_id,
                    start_time=pd.Timestamp(args.start_time),
                    windows=windows,
                    device=device,
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
    parser = argparse.ArgumentParser(description="Power forecasting training and submission export.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train wind and solar multi-horizon models.")
    train_parser.add_argument("--data-root", default=".")
    train_parser.add_argument("--artifact-dir", default="artifacts")
    train_parser.add_argument("--history-steps", type=int, default=96)
    train_parser.add_argument("--train-end", default="2025-10-31 23:45:00")
    train_parser.add_argument("--valid-start", default="2025-11-01 00:00:00")
    train_parser.add_argument("--valid-end", default="2025-12-31 23:45:00")
    train_parser.add_argument("--batch-size", type=int, default=1024)
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=2e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--early-stopping-patience", type=int, default=0)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--era5-dropout-prob", type=float, default=DEFAULT_ERA5_DROPOUT_PROB)

    export_parser = subparsers.add_parser("export", help="Export 20 submission csv files.")
    export_parser.add_argument("--data-root", default=".")
    export_parser.add_argument("--artifact-dir", default="artifacts")
    export_parser.add_argument("--output-dir", default="submission")
    export_parser.add_argument("--template-dir", default="a_pred")
    export_parser.add_argument("--start-time", default="2025-11-01 00:00:00")
    export_parser.add_argument("--windows", type=int, default=None)
    export_parser.add_argument("--submission-type", type=int, default=0)
    export_parser.add_argument("--force-cpu", action="store_true")

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


# -*- coding: utf-8 -*-
# Windows 友好的单文件训练/导出脚本
# 整体流程与原 GRU 脚本保持一致：自动识别 CSV -> 训练 -> 导出
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================
# 常量定义
# ============================================================
TIME_COL = "时间"
POWER_STATION_COL = "厂站id"
POWER_COL = "功率"
WIND_MET_COL = "测风塔id"
SOLAR_MET_COL = "光伏气象站id"
CAPACITY_COL = "装机"


# ============================================================
# 工具函数（风格与原 GRU 脚本一致）
# ============================================================
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "").lower() for c in df.columns]
    return df


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def sanitize_target(series: pd.Series) -> pd.Series:
    series = to_numeric_safe(series)
    series = series.interpolate(limit_direction="both")
    series = series.ffill().bfill().fillna(0.0)
    return series


def clean_numeric_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])
    if columns:
        df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
    return df


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c.lower() in df.columns:
            return c.lower()
    return None


def adaptive_fft_features(signal: np.ndarray, top_k: int = 3, eps: float = 1e-8) -> np.ndarray:
    """
    在最近一段信号上做 FFT，提取能量占比较高的 top_k 主频幅值。
    返回形状: [top_k]
    """
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if len(x) < 4:
        return np.zeros(top_k, dtype=np.float32)
    x = x - np.mean(x)
    fft_vals = np.fft.rfft(x)
    amp = np.abs(fft_vals).astype(np.float32)
    if len(amp) <= 1:
        return np.zeros(top_k, dtype=np.float32)
    amp = amp[1:]  # 去掉直流分量
    s = float(amp.sum())
    if s <= eps:
        return np.zeros(top_k, dtype=np.float32)
    energy = amp / (s + eps)
    k = min(top_k, len(energy))
    idx = np.argsort(energy)[-k:]
    idx = np.sort(idx)
    out = amp[idx]
    if len(out) < top_k:
        out = np.pad(out, (0, top_k - len(out)), mode="constant")
    return out.astype(np.float32)


# ============================================================
# 文件角色自动识别（与原 GRU 脚本同风格）
# ============================================================
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
            df_head = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df_head = pd.read_csv(path, nrows=0, encoding="gb18030")

        header = [str(c).strip().replace("\ufeff", "").lower() for c in df_head.columns]
        header_set = set(header)

        l_time = TIME_COL.lower()
        l_station = POWER_STATION_COL.lower()
        l_power = POWER_COL.lower()
        l_wind_met = WIND_MET_COL.lower()
        l_solar_met = SOLAR_MET_COL.lower()

        if {l_time, l_station, l_power}.issubset(header_set):
            roles["power"] = path
        elif l_wind_met in header_set and l_station in header_set and l_time not in header_set:
            roles["wind_mapping"] = path
        elif ("气象站id" in header_set or l_solar_met in header_set) and l_station in header_set and l_time not in header_set:
            roles["solar_mapping"] = path
        elif l_time in header_set and l_wind_met in header_set:
            roles["wind_features"] = path
        elif l_time in header_set and (l_solar_met in header_set or "气象站id" in header_set):
            roles["solar_features"] = path
        else:
            # 文件名兜底匹配
            if "测风塔" in path.name and "场站" in path.name:
                roles["wind_mapping"] = path
            elif "测风塔" in path.name:
                roles["wind_features"] = path
            elif ("光伏气象站" in path.name or "测光塔" in path.name) and "场站" in path.name:
                roles["solar_mapping"] = path
            elif "测光塔" in path.name or "光伏气象站" in path.name:
                roles["solar_features"] = path

    print("--- 自动识别结果 ---")
    for role, p in roles.items():
        print(f"角色 [{role}] -> 文件: {p.name}")

    required = ["power", "wind_features", "solar_features", "wind_mapping", "solar_mapping"]
    missing = [name for name in required if name not in roles]
    if missing:
        print(f"当前目录下所有CSV: {[f.name for f in all_csvs]}")
        raise FileNotFoundError(f"无法识别必要的数据文件角色: {missing}。请检查文件表头或文件名。")

    return DataFiles(**roles)


# ============================================================
# 特征工程
# ============================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df[TIME_COL.lower()]
    minutes = t.dt.hour * 60 + t.dt.minute
    quarter = minutes / (24 * 60)
    day_of_year = (t.dt.dayofyear - 1) / 366.0
    week_of_cycle = (t.dt.dayofweek * 24 * 60 + minutes) / (7 * 24 * 60)

    df["tod_sin"] = np.sin(2 * np.pi * quarter)
    df["tod_cos"] = np.cos(2 * np.pi * quarter)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year)
    df["dow_sin"] = np.sin(2 * np.pi * week_of_cycle)
    df["dow_cos"] = np.cos(2 * np.pi * week_of_cycle)
    df["month"] = t.dt.month.astype(np.float32)
    df["hour"] = t.dt.hour.astype(np.float32)
    return df


def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在基础清洗上新增风电特征：
    - 风速平方/立方
    - 风向分解为 U/V
    - 滑动统计特征
    注意：特征工程可能引入 NaN，后续会再次统一清洗。
    """
    cols = list(df.columns)

    # 风速幂次特征
    for col in cols:
        if "风速" in col:
            df[col] = to_numeric_safe(df[col])
            df[f"{col}_square"] = df[col] ** 2
            df[f"{col}_cube"] = df[col] ** 3

    # 风向分解
    for col in cols:
        if "风向" in col:
            df[col] = to_numeric_safe(df[col])
            speed_col = col.replace("风向", "风速")
            if speed_col in df.columns:
                angle_rad = np.deg2rad(df[col].fillna(0.0))
                ws = to_numeric_safe(df[speed_col]).fillna(0.0)
                df[f"{speed_col}_u"] = ws * np.cos(angle_rad)
                df[f"{speed_col}_v"] = ws * np.sin(angle_rad)

    # 主要风速列的滑动统计
    candidate_cols = [
        c for c in df.columns
        if "风速" in c and all(k not in c for k in ["square", "cube", "_u", "_v"])
    ]
    for base_col in candidate_cols[:3]:
        s = to_numeric_safe(df[base_col])
        df[f"{base_col}_ma4"] = s.rolling(4, min_periods=1).mean()
        df[f"{base_col}_ma16"] = s.rolling(16, min_periods=1).mean()
        df[f"{base_col}_diff1"] = s.diff().fillna(0.0)

    return df


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    irr = "总辐照度".lower()
    dni = "直接辐照度".lower()
    dhi = "间接辐照度".lower()
    temp = "温度".lower()
    panel_temp = "光伏电池板温度".lower()
    wind_speed = "风速".lower()
    wind_dir = "风向".lower()
    humidity = "湿度".lower()

    for src in [irr, dni, dhi, temp, panel_temp, wind_speed, wind_dir, humidity]:
        if src in df.columns:
            df[src] = to_numeric_safe(df[src])

    if irr in df.columns:
        df["is_night"] = (df[irr].fillna(0.0) <= 1.0).astype(np.float32)
    else:
        df["is_night"] = 0.0

    if irr in df.columns and temp in df.columns:
        df["irradiance_temp"] = df[irr].fillna(0.0) * df[temp].ffill().fillna(0.0)
    if irr in df.columns and panel_temp in df.columns:
        df["irradiance_panel_temp"] = df[irr].fillna(0.0) * df[panel_temp].ffill().fillna(0.0)
    if temp in df.columns and panel_temp in df.columns:
        df["panel_temp_delta"] = df[panel_temp] - df[temp]

    if wind_speed in df.columns and wind_dir in df.columns:
        angle_rad = np.deg2rad(df[wind_dir].fillna(0.0))
        ws = df[wind_speed].fillna(0.0)
        df["wind_u"] = ws * np.cos(angle_rad)
        df["wind_v"] = ws * np.sin(angle_rad)

    if irr in df.columns and dni in df.columns:
        df["dni_ratio"] = df[dni] / df[irr].replace(0.0, np.nan)
    if irr in df.columns and dhi in df.columns:
        df["dhi_ratio"] = df[dhi] / df[irr].replace(0.0, np.nan)

    for base_col in [irr, temp, panel_temp, humidity, wind_speed]:
        if base_col in df.columns:
            df[f"{base_col}_ma4"] = df[base_col].rolling(4, min_periods=1).mean()
            df[f"{base_col}_ma16"] = df[base_col].rolling(16, min_periods=1).mean()
            df[f"{base_col}_diff1"] = df[base_col].diff().fillna(0.0)

    return df


# ============================================================
# 站点样本构建
# ============================================================
@dataclass
class StationFrame:
    source: str
    met_station_id: str
    plant_id: str
    capacity: float
    frame: pd.DataFrame


def build_station_frames(root: Path) -> List[StationFrame]:
    files = detect_data_files(root)

    power_df = normalize_columns(read_csv_auto(files.power))
    wind_df = normalize_columns(read_csv_auto(files.wind_features))
    solar_df = normalize_columns(read_csv_auto(files.solar_features))
    wind_map = normalize_columns(read_csv_auto(files.wind_mapping))
    solar_map = normalize_columns(read_csv_auto(files.solar_mapping))

    l_time = TIME_COL.lower()
    l_wind_met = WIND_MET_COL.lower()
    l_station = POWER_STATION_COL.lower()
    l_power = POWER_COL.lower()
    l_capacity = CAPACITY_COL.lower()

    solar_feat_key = find_first_existing(solar_df, [SOLAR_MET_COL, "气象站id"])
    solar_map_key = find_first_existing(solar_map, [SOLAR_MET_COL, "气象站id"])
    if solar_feat_key is None or solar_map_key is None:
        raise KeyError("光伏气象表或映射表缺少气象站ID列")

    for df in [power_df, wind_df, solar_df]:
        df[l_time] = pd.to_datetime(df[l_time], errors="coerce")

    for df in [power_df, wind_df, solar_df, wind_map, solar_map]:
        for c in [l_wind_met, solar_feat_key, solar_map_key, l_station]:
            if c in df.columns:
                df[c] = df[c].astype(str)

    power_df = power_df.dropna(subset=[l_time]).sort_values(l_time)
    wind_df = wind_df.dropna(subset=[l_time]).sort_values(l_time)
    solar_df = solar_df.dropna(subset=[l_time]).sort_values(l_time)

    station_frames: List[StationFrame] = []

    # 风电站点
    for _, row in wind_map.iterrows():
        met_id = str(row[l_wind_met])
        plant_id = str(row[l_station])
        capacity = float(pd.to_numeric(row[l_capacity], errors="coerce"))
        if not np.isfinite(capacity):
            continue

        feat = wind_df[wind_df[l_wind_met] == met_id].copy()
        pwr = power_df[power_df[l_station] == plant_id].copy()
        if feat.empty or pwr.empty:
            continue

        merged = feat.merge(pwr[[l_time, l_power]], on=l_time, how="inner").sort_values(l_time)
        if merged.empty:
            continue

        merged["capacity"] = capacity
        merged["power_norm"] = sanitize_target(merged[l_power]) / max(capacity, 1e-6)
        merged = add_time_features(merged)
        merged = add_wind_features(merged)

        station_frames.append(StationFrame("wind", met_id, plant_id, capacity, merged))

    # 光伏站点
    for _, row in solar_map.iterrows():
        met_id = str(row[solar_map_key])
        plant_id = str(row[l_station])
        capacity = float(pd.to_numeric(row[l_capacity], errors="coerce"))
        if not np.isfinite(capacity):
            continue

        feat = solar_df[solar_df[solar_feat_key] == met_id].copy()
        pwr = power_df[power_df[l_station] == plant_id].copy()
        if feat.empty or pwr.empty:
            continue

        merged = feat.merge(pwr[[l_time, l_power]], on=l_time, how="inner").sort_values(l_time)
        if merged.empty:
            continue

        merged["capacity"] = capacity
        merged["power_norm"] = sanitize_target(merged[l_power]) / max(capacity, 1e-6)
        merged = add_time_features(merged)
        merged = add_solar_features(merged)

        station_frames.append(StationFrame("solar", met_id, plant_id, capacity, merged))

    return station_frames


# ============================================================
# 特征列选择与 source 数据打包
# ============================================================
def select_feature_columns(df: pd.DataFrame, source: str) -> List[str]:
    excluded = {
        TIME_COL.lower(),
        POWER_COL.lower(),
        "power_norm",
        "capacity",
        POWER_STATION_COL.lower(),
        WIND_MET_COL.lower(),
        SOLAR_MET_COL.lower(),
        "气象站id",
    }
    if source == "wind":
        excluded |= {c for c in df.columns if "风向" in c}
    else:
        excluded |= {"风向"}

    numeric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


def build_source_data(
    station_frames: List[StationFrame],
    source: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    source_frames = [s for s in station_frames if s.source == source]
    if not source_frames:
        raise ValueError(f"No station frames for source={source}")

    feature_cols = select_feature_columns(source_frames[0].frame, source)
    data: Dict[str, Dict[str, np.ndarray]] = {}

    for station in source_frames:
        df = station.frame.copy()

        # 第一轮清洗：原始列 + 特征工程列
        df = clean_numeric_frame(df, feature_cols + ["power_norm"])
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        # 特征工程后可能再次产生 NaN/inf，这里做二次修复
        df[feature_cols] = df[feature_cols].interpolate(limit_direction="both")
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)
        df["power_norm"] = sanitize_target(df["power_norm"]).clip(0.0, 1.2)

        features = df[feature_cols].to_numpy(dtype=np.float32)
        target = df["power_norm"].to_numpy(dtype=np.float32)
        timestamps = df[TIME_COL.lower()].to_numpy(dtype="datetime64[ns]")

        fft_ref_col = -1
        if source == "wind":
            # 优先使用轮毂高度风速作为 FFT 参考列，找不到则退化到任意风速列
            for i, c in enumerate(feature_cols):
                if "轮毂" in c and "风速" in c and all(x not in c for x in ["square", "cube", "_u", "_v"]):
                    fft_ref_col = i
                    break
            if fft_ref_col == -1:
                for i, c in enumerate(feature_cols):
                    if "风速" in c and all(x not in c for x in ["square", "cube", "_u", "_v"]):
                        fft_ref_col = i
                        break

        data[station.met_station_id] = {
            "features": features,
            "target": target,
            "timestamps": timestamps,
            "capacity": np.float32(station.capacity),
            "source": source,
            "fft_ref_col": fft_ref_col,
        }

    return data, feature_cols


# ============================================================
# 标准化器
# ============================================================
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
    mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = stacked.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


# ============================================================
# 数据集（训练/推理切片方式与 GRU 脚本一致）
# ============================================================
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
        fft_top_k: int = 3,
    ) -> None:
        self.items: List[Tuple[str, int]] = []
        self.source_data = source_data
        self.station_to_idx = station_to_idx
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.scaler = scaler
        self.fft_top_k = fft_top_k

        for station_id, bundle in source_data.items():
            ts = pd.to_datetime(bundle["timestamps"])
            for idx in range(history_steps, len(ts) - horizon_steps + 1):
                current = ts[idx]
                if start_time <= current <= end_time:
                    feature_slice = bundle["features"][idx - history_steps: idx]
                    target_slice = bundle["target"][idx: idx + horizon_steps]
                    if not np.isfinite(feature_slice).all() or not np.isfinite(target_slice).all():
                        continue
                    self.items.append((station_id, idx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        station_id, idx = self.items[index]
        bundle = self.source_data[station_id]
        feature_slice = bundle["features"][idx - self.history_steps: idx]
        scaled_features = (feature_slice - self.scaler["mean"]) / self.scaler["std"]

        # 仅风电分支注入 FFT 主频特征，按时间步复制后与原特征拼接
        if bundle.get("source") == "wind":
            ref_col = int(bundle.get("fft_ref_col", -1))
            if 0 <= ref_col < feature_slice.shape[1]:
                fft_feat = adaptive_fft_features(feature_slice[:, ref_col], top_k=self.fft_top_k)
                fft_feat = np.tile(fft_feat, (self.history_steps, 1))
            else:
                fft_feat = np.zeros((self.history_steps, self.fft_top_k), dtype=np.float32)
            scaled_features = np.concatenate([scaled_features, fft_feat], axis=1)

        power_history = bundle["target"][idx - self.history_steps: idx, None]
        x = np.concatenate([scaled_features, power_history], axis=1)
        y = bundle["target"][idx: idx + self.horizon_steps]
        station_idx = self.station_to_idx[station_id]

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.tensor(station_idx, dtype=torch.long),
            torch.from_numpy(y.astype(np.float32)),
        )


# ============================================================
# 混合模型（TimesBlock + Informer + BiGRU + Attention）
# ============================================================
class TimesBlock(nn.Module):
    def __init__(self, d_model: int, top_k: int = 3, kernel_size: int = 3):
        super().__init__()
        self.top_k = top_k
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.BatchNorm1d(d_model),
            )
            for _ in range(top_k)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        out = 0
        for conv in self.convs:
            out = out + conv(xt)
        out = out / self.top_k
        return out.transpose(1, 2)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class InformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, e_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads, dropout) for _ in range(e_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class HybridInformerTimesNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon_steps: int,
        station_count: int,
        hidden_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.times_block = TimesBlock(d_model=d_model, top_k=3)
        self.informer = InformerEncoder(d_model=d_model, n_heads=n_heads, e_layers=1, dropout=0.1)
        self.fuse = nn.Linear(d_model * 2, hidden_dim)
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.gru_dropout = nn.Dropout(0.1)
        self.attn = nn.MultiheadAttention(hidden_dim * 2, 4, batch_first=True)
        self.station_embedding = nn.Embedding(station_count, 16)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, horizon_steps),
        )

    def forward(self, x: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        t_feat = self.times_block(x)
        i_feat = self.informer(x)
        h = torch.cat([t_feat, i_feat], dim=-1)
        h = self.fuse(h)
        gru_out, _ = self.bigru(h)
        gru_out = self.gru_dropout(gru_out)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        summary = attn_out[:, -1, :]
        station_vec = self.station_embedding(station_idx)
        return self.head(torch.cat([summary, station_vec], dim=1))


# ============================================================
# 训练与评估（方法与 GRU 脚本保持一致）
# ============================================================
@dataclass
class TrainConfig:
    data_root: str = "."
    artifact_dir: str = "artifacts"
    history_steps: int = 96
    train_end: str = "2025-10-31 23:45:00"
    valid_start: str = "2025-11-01 00:00:00"
    valid_end: str = "2025-12-31 23:45:00"
    batch_size: int = 128
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    seed: int = 42
    num_workers: int = 0


def nmae_on_normalized(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    with torch.inference_mode():
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

    if len(train_ds) == 0:
        raise ValueError(f"No valid training windows found for source={source}, horizon_steps={horizon_steps}.")
    if len(valid_ds) == 0:
        raise ValueError(f"No valid validation windows found for source={source}, horizon_steps={horizon_steps}.")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    fft_extra = 3 if source == "wind" else 0
    model = HybridInformerTimesNet(
        input_dim=len(feature_cols) + fft_extra + 1,
        horizon_steps=horizon_steps,
        station_count=len(station_ids),
        hidden_dim=config.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_score = float("inf")
    best_state = None
    train_losses: List[float] = []
    valid_losses: List[float] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        progress = tqdm(train_loader, desc=f"{source}-{horizon_steps} epoch {epoch}", leave=False)
        for x, station_idx, y in progress:
            x = x.to(device, non_blocking=True)
            station_idx = station_idx.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x, station_idx)
                loss = nmae_on_normalized(pred, y)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected for source={source}, horizon_steps={horizon_steps}")
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            batch_loss = float(loss.detach().cpu())
            epoch_losses.append(batch_loss)
            progress.set_postfix(loss=f"{batch_loss:.5f}")

        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)
        valid_score = evaluate_model(model, valid_loader, device)
        valid_losses.append(valid_score)
        print(f"[{source}][{horizon_steps}] epoch={epoch} train_nmae={train_loss:.6f} valid_nmae={valid_score:.6f}")
        if valid_score < best_score:
            best_score = valid_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    # ---- 绘制 loss 曲线 ----
    _plot_loss_curve(
        train_losses=train_losses,
        valid_losses=valid_losses,
        best_valid=best_score,
        label=f"{source}_{horizon_steps}step",
        save_dir=artifact_dir,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": best_state,
        "source": source,
        "horizon_steps": horizon_steps,
        "history_steps": config.history_steps,
        "feature_cols": feature_cols,
        "station_ids": station_ids,
        "station_to_idx": station_to_idx,
        "scaler_mean": scaler["mean"],  # 特征标准化参数，不是混合精度的 grad_scaler
        "scaler_std": scaler["std"],
        "config": asdict(config),
        "best_valid_nmae": best_score,
        "hidden_dim": config.hidden_dim,
        "fft_extra": fft_extra,
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


def _plot_loss_curve(
    train_losses: List[float],
    valid_losses: List[float],
    best_valid: float,
    label: str,
    save_dir: Path,
) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    best_epoch = int(np.argmin(valid_losses)) + 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train NMAE")
    ax.plot(epochs, valid_losses, marker="s", label="Valid NMAE")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    ax.scatter([best_epoch], [best_valid], color="red", zorder=5,
               label=f"Best valid = {best_valid:.6f} (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NMAE")
    ax.set_title(f"Loss curve — {label}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{label}_loss_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved → {out_path}  |  best_valid_nmae = {best_valid:.6f}")


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
        seed=args.seed,
        num_workers=args.num_workers,
    )
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    station_frames = build_station_frames(Path(config.data_root))
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


# ============================================================
# 导出（预测流程与 GRU 脚本一致；最终聚合为 4 个文件）
# ============================================================
def load_artifact(path: Path, device: torch.device) -> Dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridInformerTimesNet(
        input_dim=len(payload["feature_cols"]) + int(payload.get("fft_extra", 0)) + 1,
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

    # 先解包 artifact，horizon_steps 在构建 start_indices 时就需要用到
    history_steps = int(artifact["history_steps"])
    horizon_steps = int(artifact["horizon_steps"])
    scaler_mean = artifact["scaler_mean"]
    scaler_std = artifact["scaler_std"]
    station_to_idx = artifact["station_to_idx"]
    model = artifact["model"]
    fft_extra = int(artifact.get("fft_extra", 0))

    start_indices = []
    current = start_time
    ts_to_idx = {timestamp: i for i, timestamp in enumerate(ts)}
    for _ in range(windows):
        if current not in ts_to_idx:
            raise KeyError(f"Timestamp {current} not found for station {station_id}")
        start_indices.append(ts_to_idx[current])
        # 每个窗口向后跳一个完整预测跨度（windows 个不重叠块），输出行数 = windows × horizon_steps
        current += pd.Timedelta(minutes=15)

    outputs = []
    with torch.inference_mode():
        for idx in tqdm(start_indices, desc=f"predict-{station_id}-{horizon_steps}", leave=False):
            feature_slice = features[idx - history_steps: idx]
            if len(feature_slice) < history_steps:
                raise ValueError(f"Not enough history before {ts[idx]} for station {station_id}")
            scaled_features = (feature_slice - scaler_mean) / scaler_std
            if bundle.get("source") == "wind" and fft_extra > 0:
                ref_col = int(bundle.get("fft_ref_col", -1))
                if 0 <= ref_col < feature_slice.shape[1]:
                    fft_feat = adaptive_fft_features(feature_slice[:, ref_col], top_k=fft_extra)
                    fft_feat = np.tile(fft_feat, (history_steps, 1))
                else:
                    fft_feat = np.zeros((history_steps, fft_extra), dtype=np.float32)
                scaled_features = np.concatenate([scaled_features, fft_feat], axis=1)

            power_history = target[idx - history_steps: idx, None]
            x = np.concatenate([scaled_features, power_history], axis=1)
            x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
            station_tensor = torch.tensor([station_to_idx[station_id]], dtype=torch.long, device=device)
            pred_norm = model(x_tensor, station_tensor).squeeze(0).detach().cpu().numpy()
            pred_norm = np.clip(pred_norm, 0.0, 1.2)
            pred = pred_norm * float(bundle["capacity"])
            pred = np.clip(pred, 0.0, float(bundle["capacity"]))
            outputs.append(pred[:horizon_steps])
    return np.concatenate(outputs, axis=0)


def build_submission_name(source: str, horizon_steps: int) -> str:
    prefix = "风电" if source == "wind" else "光电"
    suffix = "4h" if horizon_steps == 16 else "24h"
    return f"{prefix}_{suffix}.csv"


def run_export(args: argparse.Namespace) -> None:
    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    station_frames = build_station_frames(root)
    source_station_frames = {"wind": [], "solar": []}
    for item in station_frames:
        source_station_frames[item.source].append(item)

    # 最终仅输出 4 个聚合文件：风电/光电 × 4h/24h
    for source in ("wind", "solar"):
        source_data, _ = build_source_data(station_frames, source)
        for horizon_steps in (16, 96):
            artifact = load_artifact(Path(args.artifact_dir) / f"{source}_{horizon_steps}step.pt", device)
            rows = []
            global_id = 0
            for station in source_station_frames[source]:
                station_id = station.met_station_id
                # 若存在站点模板文件，优先按模板行数推断窗口数
                station_template_name = f"{'风电' if source == 'wind' else '光电'}_{station_id}_{'4h' if horizon_steps == 16 else '24h'}.csv"
                template_path = root / args.template_dir / station_template_name
                if template_path.exists():
                    windows = infer_sample_windows(template_path, horizon_steps)
                elif args.windows is not None:
                    windows = args.windows
                else:
                    raise FileNotFoundError(f"Template not found and --windows not set: {template_path}")

                pred = predict_for_station(
                    bundle=source_data[station_id],
                    artifact=artifact,
                    station_id=station_id,
                    start_time=pd.Timestamp(args.start_time),
                    windows=windows,
                    device=device,
                )

                for value in pred.astype(np.float32):
                    rows.append(
                        {
                            "id": global_id,
                            "pred": float(value),
                            "type": int(args.submission_type),
                        }
                    )
                    global_id += 1

            out_df = pd.DataFrame(rows, columns=["id", "pred", "type"])
            file_name = build_submission_name(source, horizon_steps)
            out_df.to_csv(output_dir / file_name, index=False, encoding="utf-8")
            print(f"Saved aggregate file: {output_dir / file_name} | rows={len(out_df)}")


# ============================================================
# 命令行入口（参数风格与 GRU 脚本一致）
# ============================================================
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
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)

    export_parser = subparsers.add_parser("export", help="Export 4 aggregated prediction csv files.")
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

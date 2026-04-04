import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from power_forecasting_itransformer_final import (
    build_source_data,
    build_station_frames,
    build_submission_name,
    load_artifact,
    predict_for_station,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Competition Docker inference entrypoint.")
    parser.add_argument("--data_root", type=str, default="/input", help="Input data directory.")
    parser.add_argument("--output_dir", type=str, default="/output", help="Output directory.")
    parser.add_argument("--artifact_dir", type=str, default="/model", help="Model artifact directory.")
    parser.add_argument("--time_start", type=str, required=True, help="Forecast start time.")
    parser.add_argument("--submission_type", type=int, default=0, help="Value for the type column.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU inference.")
    return parser.parse_args()


def validate_artifacts(artifact_dir: Path) -> None:
    required = [
        artifact_dir / "wind_16step.pt",
        artifact_dir / "wind_96step.pt",
        artifact_dir / "solar_16step.pt",
        artifact_dir / "solar_96step.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required model artifacts: {missing}")


def run_export(args: argparse.Namespace) -> Path:
    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_artifacts(artifact_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    start_time = pd.Timestamp(args.time_start)

    station_frames = build_station_frames(root)
    source_station_frames = {"wind": [], "solar": []}
    for item in station_frames:
        source_station_frames[item.source].append(item)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for source in ("wind", "solar"):
            source_data, _ = build_source_data(station_frames, source)
            for horizon_steps in (16, 96):
                artifact = load_artifact(artifact_dir / f"{source}_{horizon_steps}step.pt", device)
                for station in source_station_frames[source]:
                    pred = predict_for_station(
                        bundle=source_data[station.met_station_id],
                        artifact=artifact,
                        station_id=station.met_station_id,
                        start_time=start_time,
                        windows=1,
                        device=device,
                    )

                    file_name = build_submission_name(source, station.met_station_id, horizon_steps)
                    df = pd.DataFrame(
                        {
                            "id": np.arange(len(pred), dtype=np.int64),
                            "pred": pred.astype(np.float32),
                            "type": np.full(len(pred), args.submission_type, dtype=np.int64),
                        }
                    )
                    df.to_csv(tmpdir_path / file_name, index=False, encoding="utf-8")

        zip_path = output_dir / "predictions.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for csv_file in sorted(tmpdir_path.glob("*.csv")):
                zf.write(csv_file, arcname=csv_file.name)

    print(f"Generated prediction archive: {zip_path}")
    return zip_path


def main() -> None:
    args = parse_args()
    run_export(args)


if __name__ == "__main__":
    main()

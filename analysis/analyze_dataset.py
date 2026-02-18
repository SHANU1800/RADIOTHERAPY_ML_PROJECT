from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class FileSummary:
    file_path: str
    file_type: str
    patient_folder: str
    session_number: str = ""
    medical_id: str = ""
    date: str = ""
    time: str = ""
    sample_rate_hz: Optional[float] = None
    rows: int = 0
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    duration_s: Optional[float] = None
    volume_min: Optional[float] = None
    volume_max: Optional[float] = None
    volume_mean: Optional[float] = None
    volume_std: Optional[float] = None
    balloon_status_counts: Dict[str, int] = field(default_factory=dict)
    patient_switch_counts: Dict[str, int] = field(default_factory=dict)
    gating_mode_counts: Dict[str, int] = field(default_factory=dict)


def normalize_flag(value: str) -> Optional[str]:
    cleaned = value.strip()
    if not cleaned or cleaned in {"-", "-", "-"}:
        return None
    return cleaned


def increment_count(counter: Dict[str, int], key: Optional[str]) -> None:
    if key is None:
        return
    counter[key] = counter.get(key, 0) + 1


def parse_header(lines: Iterable[str]) -> Tuple[Dict[str, str], List[str]]:
    meta: Dict[str, str] = {}
    remaining_lines: List[str] = []
    header_done = False
    for line in lines:
        if not header_done:
            stripped = line.strip()
            if stripped == "HeaderEnd":
                header_done = True
                continue
            if ":" in stripped and not stripped.startswith("["):
                key, value = stripped.split(":", 1)
                meta[key.strip()] = value.strip()
            continue
        remaining_lines.append(line)
    return meta, remaining_lines


def estimate_sample_rate(times: List[float]) -> Optional[float]:
    if len(times) < 2:
        return None
    diffs = [t2 - t1 for t1, t2 in zip(times, times[1:]) if t2 > t1]
    if not diffs:
        return None
    avg_dt = sum(diffs) / len(diffs)
    return round(1.0 / avg_dt, 3) if avg_dt > 0 else None


def analyze_txt_or_dat_file(file_path: Path, patient_folder: str, file_type: str = "txt") -> FileSummary:
    stats = RunningStats()
    balloon_counts: Dict[str, int] = {}
    patient_switch_counts: Dict[str, int] = {}
    gating_mode_counts: Dict[str, int] = {}
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    time_samples: List[float] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        meta, data_lines = parse_header(handle)

    for line in data_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("Session Time"):
            continue
        parts = [p.strip() for p in stripped.split(";")]
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            v = float(parts[1])
        except ValueError:
            continue

        time_start = t if time_start is None else min(time_start, t)
        time_end = t if time_end is None else max(time_end, t)
        if len(time_samples) < 100:
            time_samples.append(t)

        stats.update(v)

        if len(parts) > 2:
            increment_count(balloon_counts, parts[2] or None)
        if len(parts) > 3:
            increment_count(patient_switch_counts, parts[3] or None)
        if len(parts) > 4:
            increment_count(gating_mode_counts, normalize_flag(parts[4]))

    sample_rate = None
    if "Count Frequency" in meta:
        try:
            sample_rate = float(meta["Count Frequency"])
        except ValueError:
            sample_rate = None
    if sample_rate is None:
        sample_rate = estimate_sample_rate(time_samples)

    summary = FileSummary(
        file_path=str(file_path),
        file_type=file_type,
        patient_folder=patient_folder,
        session_number=meta.get("Session Number", ""),
        medical_id=meta.get("Medical ID", ""),
        date=meta.get("Date", ""),
        time=meta.get("Time", ""),
        sample_rate_hz=sample_rate,
        rows=stats.count,
        time_start=time_start,
        time_end=time_end,
        duration_s=(time_end - time_start) if time_start is not None and time_end is not None else None,
        volume_min=stats.min_value,
        volume_max=stats.max_value,
        volume_mean=stats.mean if stats.count else None,
        volume_std=stats.std if stats.count else None,
        balloon_status_counts=balloon_counts,
        patient_switch_counts=patient_switch_counts,
        gating_mode_counts=gating_mode_counts,
    )
    return summary


def analyze_csv_file(file_path: Path, patient_folder: str) -> FileSummary:
    stats = RunningStats()
    balloon_counts: Dict[str, int] = {}
    patient_switch_counts: Dict[str, int] = {}
    gating_mode_counts: Dict[str, int] = {}
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    time_samples: List[float] = []

    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            try:
                t = float(row.get("Session Time", ""))
                v = float(row.get("Volume (liters)", ""))
            except ValueError:
                continue

            time_start = t if time_start is None else min(time_start, t)
            time_end = t if time_end is None else max(time_end, t)
            if len(time_samples) < 100:
                time_samples.append(t)

            stats.update(v)

            increment_count(balloon_counts, row.get("Balloon Valve Status") or None)
            increment_count(patient_switch_counts, row.get("Patient Switch") or None)
            increment_count(gating_mode_counts, normalize_flag(row.get("Gating Mode", "")))

    sample_rate = estimate_sample_rate(time_samples)

    summary = FileSummary(
        file_path=str(file_path),
        file_type="csv",
        patient_folder=patient_folder,
        sample_rate_hz=sample_rate,
        rows=stats.count,
        time_start=time_start,
        time_end=time_end,
        duration_s=(time_end - time_start) if time_start is not None and time_end is not None else None,
        volume_min=stats.min_value,
        volume_max=stats.max_value,
        volume_mean=stats.mean if stats.count else None,
        volume_std=stats.std if stats.count else None,
        balloon_status_counts=balloon_counts,
        patient_switch_counts=patient_switch_counts,
        gating_mode_counts=gating_mode_counts,
    )
    return summary


def summarize_dataset(dataset_dir: Path) -> Tuple[List[FileSummary], List[Dict[str, str]]]:
    summaries: List[FileSummary] = []
    other_files: List[Dict[str, str]] = []

    for patient_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        patient_folder = patient_dir.name
        for file_path in sorted(patient_dir.rglob("*")):
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() == ".txt":
                summaries.append(analyze_txt_or_dat_file(file_path, patient_folder, "txt"))
            elif file_path.suffix.lower() == ".dat":
                summaries.append(analyze_txt_or_dat_file(file_path, patient_folder, "dat"))
            elif file_path.suffix.lower() == ".csv":
                summaries.append(analyze_csv_file(file_path, patient_folder))
            else:
                other_files.append(
                    {
                        "file_path": str(file_path),
                        "patient_folder": patient_folder,
                        "extension": file_path.suffix.lower(),
                        "size_bytes": str(file_path.stat().st_size),
                    }
                )

    return summaries, other_files


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    dataset_dir = DATASET_DIR
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    summaries, other_files = summarize_dataset(dataset_dir)

    file_rows: List[Dict[str, str]] = []
    patient_agg: Dict[str, Dict[str, float]] = {}

    for summary in summaries:
        file_rows.append(
            {
                "file_path": summary.file_path,
                "file_type": summary.file_type,
                "patient_folder": summary.patient_folder,
                "session_number": summary.session_number,
                "medical_id": summary.medical_id,
                "date": summary.date,
                "time": summary.time,
                "sample_rate_hz": "" if summary.sample_rate_hz is None else str(summary.sample_rate_hz),
                "rows": str(summary.rows),
                "time_start": "" if summary.time_start is None else str(summary.time_start),
                "time_end": "" if summary.time_end is None else str(summary.time_end),
                "duration_s": "" if summary.duration_s is None else str(summary.duration_s),
                "volume_min": "" if summary.volume_min is None else str(summary.volume_min),
                "volume_max": "" if summary.volume_max is None else str(summary.volume_max),
                "volume_mean": "" if summary.volume_mean is None else str(summary.volume_mean),
                "volume_std": "" if summary.volume_std is None else str(summary.volume_std),
                "balloon_status_counts": json.dumps(summary.balloon_status_counts, ensure_ascii=False),
                "patient_switch_counts": json.dumps(summary.patient_switch_counts, ensure_ascii=False),
                "gating_mode_counts": json.dumps(summary.gating_mode_counts, ensure_ascii=False),
            }
        )

        agg = patient_agg.setdefault(
            summary.patient_folder,
            {"files": 0.0, "rows": 0.0, "duration_s": 0.0},
        )
        agg["files"] += 1
        agg["rows"] += summary.rows
        if summary.duration_s:
            agg["duration_s"] += summary.duration_s

    patient_rows = [
        {
            "patient_folder": patient,
            "files": str(int(values["files"])),
            "rows": str(int(values["rows"])),
            "duration_s": str(values["duration_s"]),
        }
        for patient, values in sorted(patient_agg.items())
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "file_summary.csv", file_rows)
    write_csv(OUTPUT_DIR / "patient_summary.csv", patient_rows)
    write_csv(OUTPUT_DIR / "other_files.csv", other_files)

    print(f"Wrote {len(file_rows)} file summaries to {OUTPUT_DIR / 'file_summary.csv'}")
    print(f"Wrote {len(patient_rows)} patient summaries to {OUTPUT_DIR / 'patient_summary.csv'}")
    print(f"Wrote {len(other_files)} other files to {OUTPUT_DIR / 'other_files.csv'}")


if __name__ == "__main__":
    main()

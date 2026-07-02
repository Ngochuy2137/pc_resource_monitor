#!/usr/bin/env python3
"""
Log CPU and GPU usage to an Excel file on Jetson (or any Linux host with psutil).

CPU sampling follows main_controller/observer.py:
  psutil.cpu_percent(interval=0, percpu=True)
  psutil.virtual_memory().used

GPU load is read from Jetson sysfs (value / 10 = percent).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("Install dependencies: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1)

try:
    from openpyxl import Workbook
except ImportError:
    print("Install dependencies: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1)

DEFAULT_GPU_LOAD_PATHS = (
    "/sys/devices/platform/bus@0/17000000.gpu/load",
    "/sys/class/devfreq/17000000.gpu/device/load",
)


def find_gpu_load_path() -> Path | None:
    for path_str in DEFAULT_GPU_LOAD_PATHS:
        path = Path(path_str)
        if path.is_file():
            return path

    devfreq = Path("/sys/class/devfreq")
    if devfreq.is_dir():
        for entry in devfreq.iterdir():
            name_file = entry / "device/of_node/name"
            if name_file.is_file():
                name = name_file.read_text(encoding="utf-8", errors="ignore").strip("\x00")
                if name == "gpu":
                    load_file = entry / "device/load"
                    if load_file.is_file():
                        return load_file
    return None


def read_gpu_percent(gpu_load_path: Path | None) -> float | None:
    if gpu_load_path is None:
        return None
    try:
        raw = gpu_load_path.read_text(encoding="utf-8").strip()
        return float(raw) / 10.0
    except (OSError, ValueError):
        return None


def get_cpu_per_core() -> list[float]:
    return list(psutil.cpu_percent(interval=0, percpu=True))


def get_system_ram_gb() -> float:
    return psutil.virtual_memory().used / (1024**3)


def count_cores_above(per_core: list[float], threshold: float) -> int:
    return sum(1 for value in per_core if value > threshold)


def find_pids_by_cmdline(fragment: str) -> list[int]:
    fragment_lower = fragment.lower()
    own_pid = os.getpid()
    pids: list[int] = []
    for proc in psutil.process_iter(["pid", "cmdline", "exe"]):
        pid = proc.info["pid"]
        if pid == own_pid:
            continue
        cmdline = " ".join(proc.info["cmdline"] or [])
        if fragment_lower not in cmdline.lower():
            continue
        exe = (proc.info["exe"] or "").lower()
        if exe.endswith("/bash") or exe.endswith("/sh"):
            continue
        pids.append(pid)
    return pids


def normalize_ros2_node_name(node_name: str) -> str:
    return node_name.strip().lstrip("/")


def ros2_node_to_process_match(node_name: str) -> str:
    """ROS2 passes the logical node name in process cmdline as __node:=NAME."""
    return f"__node:={normalize_ros2_node_name(node_name)}"


def find_pids_by_ros2_node(node_name: str) -> list[int]:
    return find_pids_by_cmdline(ros2_node_to_process_match(node_name))


def normalize_target_processes(names: list[str]) -> list[str]:
    return [normalize_ros2_node_name(name) for name in names if name.strip()]


def build_columns(threshold: float, target_processes: list[str]) -> list[str]:
    cores_col = f"system_cores_above_{threshold:g}pct"
    columns = [
        "timestamp",
        "system_cpu_avg_pct",
        "system_gpu_pct",
        "system_ram_gb",
        cores_col,
    ]
    for name in target_processes:
        columns.append(f"{name}_cpu_avg_pct")
        columns.append(f"{name}_ram_gb")
    return columns


def sample_process_resource(
    pids: list[int],
    process_cache: dict[int, psutil.Process],
) -> tuple[float, float]:
    cpu_total = 0.0
    ram_gb = 0.0

    for pid in pids:
        try:
            proc = process_cache.get(pid)
            if proc is None:
                proc = psutil.Process(pid)
                process_cache[pid] = proc
                proc.cpu_percent(interval=None)
                continue
            cpu_total += proc.cpu_percent(interval=None)
            ram_gb += proc.memory_info().rss / (1024**3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_cache.pop(pid, None)
            continue
    return cpu_total, ram_gb


def prime_process_cpu_counters(
    pids: list[int],
    process_cache: dict[int, psutil.Process],
) -> None:
    for pid in pids:
        try:
            proc = process_cache.get(pid)
            if proc is None:
                proc = psutil.Process(pid)
                process_cache[pid] = proc
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_cache.pop(pid, None)
            continue


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / f"cpu_gpu_monitor_{stamp}.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor CPU/GPU usage and save samples to an Excel file."
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Sampling rate in Hz (default: 10)",
    )
    parser.add_argument(
        "-x",
        "--threshold",
        type=float,
        default=80.0,
        help="CPU core usage threshold in percent for the core-count column (default: 80)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds (0 = until Ctrl+C, default: 0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .xlsx path (default: cpu_gpu_monitor_<timestamp>.xlsx in package dir)",
    )
    parser.add_argument(
        "--target-processes",
        dest="target_processes",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "ROS2 node name(s) to track. Example: "
            "--target-processes slam_toolbox_localization slam_toolbox_mapping. "
            "PID is resolved from process cmdline __node:=NAME"
        ),
    )
    return parser.parse_args()


def round_cell(column: str, value: object) -> object:
    if column == "timestamp":
        return value
    if column.endswith("_pct"):
        return round(float(value), 2)
    if column.endswith("_gb"):
        return round(float(value), 3)
    if column.startswith("system_cores_above_"):
        return int(value)
    return value


def build_summary_row(columns: list[str], rows: list[dict[str, object]]) -> list[object]:
    summary: list[object] = []
    for column in columns:
        if column == "timestamp":
            summary.append("SUMMARY")
            continue

        values = [row[column] for row in rows]
        if column == "system_gpu_pct":
            gpu_values = [float(value) for value in values if value is not None]
            summary.append(
                round(statistics.fmean(gpu_values), 2) if gpu_values else None
            )
        elif column.startswith("system_cores_above_"):
            summary.append(sum(int(value) for value in values))
        elif column.endswith("_pct") or column.endswith("_gb"):
            summary.append(
                round(statistics.fmean(float(value) for value in values), 3)
                if column.endswith("_gb")
                else round(statistics.fmean(float(value) for value in values), 2)
            )
        else:
            summary.append(None)
    return summary


def main() -> int:
    args = parse_args()
    if args.hz <= 0:
        print("--hz must be > 0", file=sys.stderr)
        return 1
    if not 0 <= args.threshold <= 100:
        print("--threshold must be between 0 and 100", file=sys.stderr)
        return 1

    target_processes = (
        normalize_target_processes(args.target_processes)
        if args.target_processes
        else []
    )
    columns = build_columns(args.threshold, target_processes)
    system_cores_col = f"system_cores_above_{args.threshold:g}pct"

    interval = 1.0 / args.hz
    output_path = args.output or default_output_path()
    gpu_load_path = find_gpu_load_path()
    process_cache: dict[int, psutil.Process] = {}

    if gpu_load_path is None:
        print("Warning: GPU load sysfs not found; GPU column will be empty.", file=sys.stderr)
    else:
        print(f"GPU load path: {gpu_load_path}")

    psutil.cpu_percent(interval=0, percpu=True)
    for name in target_processes:
        prime_process_cpu_counters(find_pids_by_ros2_node(name), process_cache)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "monitor"
    duration_text = "until Ctrl+C" if args.duration <= 0 else f"{args.duration:g}s"
    sheet.append(
        [
            "RUN_PARAMS",
            f"--hz={args.hz:g}",
            f"--threshold={args.threshold:g}",
            f"duration={duration_text}",
            (
                "--target-processes=" + ",".join(target_processes)
                if target_processes
                else ""
            ),
        ]
    )
    sheet.append(columns)

    rows: list[dict[str, object]] = []
    start = time.monotonic()
    next_sample = start
    sample_count = 0

    print(
        f"Sampling at {args.hz:g} Hz, threshold={args.threshold:g}%%, "
        f"output={output_path}"
    )
    for name in target_processes:
        pids = find_pids_by_ros2_node(name)
        if pids:
            print(
                f"Tracking '/{name}': "
                f"cmdline match '{ros2_node_to_process_match(name)}', PIDs={pids}"
            )
        else:
            print(
                f"Warning: no process matched '/{name}'; "
                "its columns will stay empty until a match appears.",
                file=sys.stderr,
            )

    try:
        while True:
            now = time.monotonic()
            if args.duration > 0 and (now - start) >= args.duration:
                break

            if now < next_sample:
                time.sleep(min(0.001, next_sample - now))
                continue

            per_core = get_cpu_per_core()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            row: dict[str, object] = {
                "timestamp": timestamp,
                "system_cpu_avg_pct": statistics.fmean(per_core) if per_core else 0.0,
                "system_gpu_pct": read_gpu_percent(gpu_load_path),
                "system_ram_gb": get_system_ram_gb(),
                system_cores_col: count_cores_above(per_core, args.threshold),
            }

            for name in target_processes:
                pids = find_pids_by_ros2_node(name)
                cpu_pct, ram_gb = sample_process_resource(pids, process_cache)
                row[f"{name}_cpu_avg_pct"] = cpu_pct
                row[f"{name}_ram_gb"] = ram_gb

            rows.append(row)
            sample_count += 1
            next_sample += interval

            if sample_count % int(max(args.hz, 1)) == 0:
                gpu_pct = row["system_gpu_pct"]
                gpu_text = "n/a" if gpu_pct is None else f"{float(gpu_pct):.1f}"
                line = (
                    f"[{timestamp}] system_cpu={float(row['system_cpu_avg_pct']):.1f}% "
                    f"system_gpu={gpu_text}% "
                    f"system_ram={float(row['system_ram_gb']):.3f}GB "
                    f"{system_cores_col}={row[system_cores_col]}"
                )
                for name in target_processes:
                    line += (
                        f" {name}_cpu={float(row[f'{name}_cpu_avg_pct']):.1f}%"
                        f" {name}_ram={float(row[f'{name}_ram_gb']):.3f}GB"
                    )
                print(line)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    for row in rows:
        sheet.append([round_cell(column, row[column]) for column in columns])

    if rows:
        sheet.append(build_summary_row(columns, rows))

    workbook.save(output_path)
    print(f"Saved {len(rows)} samples to {output_path}")

    if rows:
        summary_row = build_summary_row(columns, rows)
        summary_map = dict(zip(columns, summary_row))
        print(
            "Summary row: "
            f"system_cpu_avg_pct={summary_map['system_cpu_avg_pct']}% "
            f"system_ram_gb={summary_map['system_ram_gb']} "
            f"{system_cores_col}={summary_map[system_cores_col]}"
        )
        for name in target_processes:
            print(
                f"  {name}: cpu_avg_pct={summary_map[f'{name}_cpu_avg_pct']}% "
                f"ram_gb={summary_map[f'{name}_ram_gb']}"
            )
        gpu_values = [
            float(row["system_gpu_pct"])
            for row in rows
            if row["system_gpu_pct"] is not None
        ]
        if gpu_values:
            print(
                f"System GPU stats: min={min(gpu_values):.1f}% "
                f"max={max(gpu_values):.1f}% "
                f"avg={statistics.fmean(gpu_values):.1f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

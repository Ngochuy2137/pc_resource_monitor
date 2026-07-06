# Jetson CPU/GPU Monitor

Log system and per-process resource usage to an Excel file.

CPU/RAM use `psutil`. GPU load on Jetson is read from sysfs (`value / 10` = percent, system-wide only).

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 monitor.py --hz 50 \
  --threshold 90 \
  --duration 60 \
  --target-processes my_node another_node \
  -o output.xlsx
```

Node names may be passed with or without a leading `/`. Stop with Ctrl+C when `--duration 0` (default).

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--hz` | `10` | Sampling rate (Hz) |
| `-x`, `--threshold` | `80` | Count CPU cores above this usage (%) |
| `--duration` | `0` | Run time in seconds (`0` = until Ctrl+C) |
| `-o`, `--output` | auto timestamp | Output `.xlsx` path |
| `--target-processes` | _(none)_ | ROS2 node name(s) to track (CPU + RAM each) |

## Excel output

| Row | Content |
|-----|---------|
| 1 | `RUN_PARAMS` — command settings used for the run |
| 2 | Column headers |
| 3…n | Samples |
| Last | `SUMMARY` — average for `*_pct` / `*_gb`, sum for `system_cores_above_*` |

### System columns

| Column | Description |
|--------|-------------|
| `timestamp` | Local time with milliseconds |
| `system_cpu_avg_pct` | Average CPU across all cores |
| `system_gpu_pct` | GPU load (%) |
| `system_ram_gb` | System RAM used (GB) |
| `system_cores_above_{x}pct` | Cores with usage > `x` |

### Per target process

For each name in `--target-processes`:

| Column | Description |
|--------|-------------|
| `{name}_cpu_avg_pct` | Process CPU (%) on the same scale as `system_cpu_avg_pct` |
| `{name}_ram_gb` | Process RSS (GB) |

PIDs are found by matching `__node:=<name>` in the process cmdline.

Target process CPU is `psutil` process utilization divided by `cpu_count`, so it is
comparable to `system_cpu_avg_pct` (average load across all cores). GPU work is not
included in process CPU.

## GPU stress (optional)

```bash
python3 gpu_stress.py
```

Requires CUDA `nvcc` for the bundled CUDA utility.

## Quick test

```bash
./run_test.sh
```

Runs GPU stress in the background and records 15 seconds at 10 Hz.

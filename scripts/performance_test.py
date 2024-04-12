#!/usr/bin/env python3

import argparse
import ast
import os
import re
from statistics import mean
from subprocess import run
from typing import Dict, List
from warnings import warn

import matplotlib.pyplot as plt

# NOTE  I rely on these names being prefixed with either "CPU_" or
#       "GPU_" to filter based on the hardware type.
# NOTE  This is ranked from slowest to fastest. Yes, the C++ scan is
#       slower than the naive version. Embarrassing!
COMMAND_LINE_SCAN_TYPES = [
    "CPU_StdSerial",
    "CPU_Serial",
    "CPU_Parallel",
    "CPU_SimulateOptimalButIncorrect",
    "GPU_NaiveHierarchical",
    "GPU_OptimizedHierarchical",
    "GPU_OurDecoupledLookback",
    "GPU_NvidiaDecoupledLookback",
    "GPU_SimulateOptimalButIncorrect",
]
COMMAND_LINE_INPUT_SIZES = [
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    200_000_000,
    300_000_000,
    400_000_000,
    500_000_000,
    600_000_000,
    700_000_000,
    800_000_000,
    900_000_000,
    1_000_000_000,
]


def get_plot_colour_and_linestyle(scan_type: str):
    """Return the Matplotlib colour and linestyle for a scan type."""
    return {
        "CPU_StdSerial": ("lightsteelblue", "solid"),
        "CPU_Serial": ("tab:blue", "solid"),
        "CPU_Parallel": ("tab:purple", "solid"),
        "CPU_SimulateOptimalButIncorrect": ("tab:cyan", "dashed"),
        "GPU_NaiveHierarchical": ("tab:red", "solid"),
        "GPU_OptimizedHierarchical": ("tab:orange", "solid"),
        "GPU_OurDecoupledLookback": ("yellow", "solid"),
        "GPU_NvidiaDecoupledLookback": ("tab:green", "solid"),
        "GPU_SimulateOptimalButIncorrect": ("lime", "dashed"),
    }.get(scan_type, ("grey", "dotted"))


def write_result_to_cache(path: str, table: dict[str, dict[int, list[int]]]):
    if os.path.exists(path):
        warn(f"overwriting path {os.path.abspath(path)}")
    with open(path, "w") as f:
        f.write(str(table))


def read_result_from_cache(path: str) -> dict[str, dict[int, list[int]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"couldn't open {os.path.abspath(path)}")
    with open(path) as f:
        text = f.read()
    return ast.literal_eval(text)


def chdir_to_top_level():
    out = run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        timeout=60,
        text=True,
    )
    path = out.stdout.strip()  # Strip whitespace from path (e.g. trailing \n)
    os.chdir(path)


def run_and_time_main(
    executable: str, scan_type: str, size: int, repeats: int, debug_mode: bool, check_output: bool
) -> list[float]:
    """Time the implementation"""
    assert scan_type in COMMAND_LINE_SCAN_TYPES
    assert size in range(1, 1_000_000_000 + 1)
    pattern = re.compile(r"@@@ Elapsed time [(]sec[)]: (\d+[.]\d+)")
    out = run(
        [
            f"./{executable}",
            "--type",
            scan_type,
            "--size",
            f"{size}",
            "--repeats",
            f"{repeats}",
            # Add optional arguments
            *(["--debug"] if debug_mode else []),
            *(["--check"] if check_output else []),
        ],
        capture_output=True,
        timeout=60,
        text=True,
    )
    print(f"{out.stdout.splitlines()}")
    if out.returncode:
        warn(f"{out.stderr.strip()}")
    lines = [re.match(pattern, s) for s in out.stdout.split("\n")]
    times = [float(m.group(1)) for m in lines if m is not None]
    return times


def postprocess_experiment_data(table: dict[str, dict[int, list[int]]]):
    assert all(k in COMMAND_LINE_SCAN_TYPES for k in table.keys())
    print(f"Raw data: {table}")
    avg_table = {
        scan_type: {size: mean(data) for size, data in data_by_size.items()}
        for scan_type, data_by_size in table.items()
    }
    print(f"Plotted data: {avg_table}")
    return avg_table


def plot_timings(avg_table: dict[str, dict[int, float]]):
    plt.figure("Performance Timing for Inclusive Scan Algorithms")
    plt.title("Performance Timing for Inclusive Scan Algorithms")

    for key, data_by_size in avg_table.items():
        colour, linestyle = get_plot_colour_and_linestyle(key)
        plt.plot(
            list(data_by_size.keys()),
            list(data_by_size.values()),
            label=key,
            color=colour,
            linestyle=linestyle,
        )

    plt.legend()
    plt.xlabel("Input Size")
    plt.ylabel("Time [seconds]")

    plt.tight_layout()
    plt.savefig("performance-timings", dpi=300, transparent=True)


def plot_gpu_timings(avg_table: dict[str, dict[int, float]]):
    plt.figure("Performance Timing for Inclusive Scan Algorithms (GPU only)")
    plt.title("Performance Timing for Inclusive Scan Algorithms (GPU only)")

    gpu_avg_table = {
        scan_type: data_by_size for scan_type, data_by_size in avg_table.items() if scan_type.startswith("GPU_")
    }

    for key, data_by_size in gpu_avg_table.items():
        colour, linestyle = get_plot_colour_and_linestyle(key)
        plt.plot(
            list(data_by_size.keys()),
            list(data_by_size.values()),
            label=key,
            color=colour,
            linestyle=linestyle,
        )

    plt.legend()
    plt.xlabel("Input Size")
    plt.ylabel("Time [seconds]")

    plt.tight_layout()
    plt.savefig("performance-timings-gpu-only", dpi=300, transparent=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompile", action="store_true", help="Recompile the executable")
    parser.add_argument("--cpu-only", action="store_true", help="Run only the CPU algorithms")
    parser.add_argument("--gpu-only", action="store_true", help="Run only the GPU algorithms")
    parser.add_argument("--repeats", type=int, default=10, help="Number of times the test is repeated")
    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use a cached version of the results. Overrides all else",
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    parser.add_argument("--check", "-c", action="store_true", help="Check output")
    parser.add_argument("--executable", default="main", help="Name of executable relative to current working directory")
    args = parser.parse_args()

    global COMMAND_LINE_SCAN_TYPES
    assert not (args.gpu_only and args.cpu_only), "choose GPU-only, CPU-only, or neither; but not both!"
    if args.cpu_only:
        COMMAND_LINE_SCAN_TYPES = [x for x in COMMAND_LINE_SCAN_TYPES if x.startswith("CPU_")]
    if args.gpu_only:
        COMMAND_LINE_SCAN_TYPES = [x for x in COMMAND_LINE_SCAN_TYPES if x.startswith("GPU_")]

    repeats, debug_mode, check_output = args.repeats, args.debug, args.check

    chdir_to_top_level()

    # Optionally recompile the executable
    if args.recompile:
        run("make clean".split())
        run("make")

    table = {key: {size: [] for size in COMMAND_LINE_INPUT_SIZES} for key in COMMAND_LINE_SCAN_TYPES}

    if args.use_cached:
        table = read_result_from_cache("cache.txt")
    else:
        for key in table:
            for size in table[key]:
                table[key][size] = run_and_time_main(args.executable, key, size, repeats, debug_mode, check_output)
        write_result_to_cache("cache.txt", table)

    avg_table = postprocess_experiment_data(table)

    plot_timings(avg_table)

    # Plot GPU timing separately if available!
    if not args.cpu_only:
        plot_gpu_timings(avg_table)


if __name__ == "__main__":
    main()

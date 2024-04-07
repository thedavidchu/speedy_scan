import argparse
import os
import re
from subprocess import run
from typing import Dict, List
from statistics import mean
from warnings import warn

import matplotlib.pyplot as plt


# NOTE  I rely on these names being prefixed with either "CPU_" or
#       "GPU_" to filter based on the hardware type.
COMMAND_LINE_SCAN_TYPES = [
    "CPU_SerialBaseline",
    "CPU_ParallelBaseline",
    "CPU_SimulateOptimalButIncorrect",
    "GPU_NaiveHierarchical",
    "GPU_OptimizedBaseline",
    "GPU_OurDecoupledLookback",
    "GPU_NvidiaDecoupledLookback",
    "GPU_SimulateOptimalButIncorrect",
]
COMMAND_LINE_INPUT_SIZES = [
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
]


def get_plot_colour(scan_type: str):
    """Return the Matplotlib colour for a scan type."""
    return {
        "CPU_SerialBaseline": ("tab:blue", "solid"),
        "CPU_ParallelBaseline": ("tab:purple", "solid"),
        "CPU_SimulateOptimalButIncorrect": ("tab:cyan", "dashed"),

        "GPU_NaiveHierarchical": ("tab:red", "solid"),
        "GPU_OptimizedBaseline": ("tab:orange", "solid"),
        "GPU_OurDecoupledLookback": ("yellow", "solid"),
        "GPU_NvidiaDecoupledLookback": ("tab:green", "solid"),
        "GPU_SimulateOptimalButIncorrect": ("lime", "dashed"),
    }.get(scan_type, ("grey", "dotted"))


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
    scan_type: str, size: int, repeats: int, debug_mode: bool, check_output: bool
) -> List[float]:
    """Time the implementation"""
    assert scan_type in COMMAND_LINE_SCAN_TYPES
    assert size in range(1, 1_000_000_000 + 1)
    pattern = re.compile(r"@@@ Elapsed time [(]sec[)]: (\d+[.]\d+)")
    out = run(
        [
            "main",
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


def postprocess_experiment_data(table: Dict[str, Dict[int, List[int]]]):
    assert all(k in COMMAND_LINE_SCAN_TYPES for k in table.keys())
    print(f"Raw data: {table}")
    avg_table = {
        scan_type: {size: mean(data) for size, data in data_by_size.items()}
        for scan_type, data_by_size in table.items()
    }
    print(f"Plotted data: {avg_table}")
    return avg_table


def plot_timings(avg_table: Dict[str, Dict[int, float]]):
    plt.figure(f"Performance Timing for Inclusive Scan Algorithms")
    plt.title(f"Performance Timing for Inclusive Scan Algorithms")

    for key, data_by_size in avg_table.items():
        colour, linestyle = get_plot_colour(key)
        plt.plot(data_by_size.keys(), data_by_size.values(), label=key, color=colour, linestyle=linestyle)

    plt.legend()
    plt.xlabel("Input Size")
    plt.ylabel("Time ")

    plt.savefig(f"performance-timings")


def plot_gpu_timings(avg_table: Dict[str, Dict[int, float]]):
    plt.figure(f"Performance Timing for Inclusive Scan Algorithms (GPU only)")
    plt.title(f"Performance Timing for Inclusive Scan Algorithms (GPU only)")

    gpu_avg_table = {
        scan_type: data_by_size
        for scan_type, data_by_size in avg_table.items()
        if scan_type.startswith("GPU_")
    }

    for key, data_by_size in gpu_avg_table.items():
        plt.plot(data_by_size.keys(), data_by_size.values(), label=key)

    plt.legend()
    plt.xlabel("Input Size")
    plt.ylabel("Time ")

    plt.savefig(f"performance-timings-gpu-only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompile", action="store_true", help="Recompile the executable"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run only the CPU algorithms"
    )
    parser.add_argument(
        "--gpu-only", action="store_true", help="Run only the GPU algorithms"
    )
    parser.add_argument(
        "--repeats", type=int, default=10, help="Number of times the test is repeated"
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    parser.add_argument("--check", "-c", action="store_true", help="Check output")
    args = parser.parse_args()

    global COMMAND_LINE_SCAN_TYPES
    assert not (
        args.gpu_only and args.cpu_only
    ), "choose GPU-only, CPU-only, or neither; but not both!"
    if args.cpu_only:
        COMMAND_LINE_SCAN_TYPES = [
            x for x in COMMAND_LINE_SCAN_TYPES if x.startswith("CPU_")
        ]
    if args.gpu_only:
        COMMAND_LINE_SCAN_TYPES = [
            x for x in COMMAND_LINE_SCAN_TYPES if x.startswith("GPU_")
        ]

    repeats, debug_mode, check_output = args.repeats, args.debug, args.check

    chdir_to_top_level()

    # Optionally recompile the executable
    if args.recompile:
        run("make clean".split())
        run("make")

    table = {
        key: {size: [] for size in COMMAND_LINE_INPUT_SIZES}
        for key in COMMAND_LINE_SCAN_TYPES
    }

    for key in table:
        for size in table[key]:
            table[key][size] = run_and_time_main(
                key, size, repeats, debug_mode, check_output
            )

    avg_table = postprocess_experiment_data(table)

    plot_timings(avg_table)
    plot_gpu_timings(avg_table)


if __name__ == "__main__":
    main()

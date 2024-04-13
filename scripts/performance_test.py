#!/usr/bin/env python3

import argparse
import ast
import os
import re
from statistics import mean
from subprocess import run
from typing import Optional
from warnings import warn

import pandas as pd
stevens_implementation = False
if stevens_implementation:
    import seaborn as sns
from matplotlib import pyplot as plt

# NOTE  I rely on these names being prefixed with either "CPU_" or
#       "GPU_" to filter based on the hardware type.
# NOTE  This is ranked from slowest to fastest. Yes, the C++ scan is
#       slower than the naive version. Embarrassing!
COMMAND_LINE_SCAN_TYPES = [
    "CPU_StdSerial",
    "CPU_Serial",
    "CPU_Parallel",
    "CPU_MemoryCopy",
    "GPU_NaiveHierarchical",
    "GPU_OptimizedHierarchical",
    "GPU_OurDecoupledLookback",
    "GPU_NvidiaDecoupledLookback",
    "GPU_CUBSimplified",
    "GPU_MemoryCopy",
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
        "CPU_StdSerial": ("CPU: std::inclusive_scan", "lightsteelblue", "solid"),
        "CPU_Serial": ("CPU: Serial", "tab:blue", "solid"),
        "CPU_Parallel": ("CPU: Parallel", "tab:purple", "solid"),
        "CPU_MemoryCopy": ("CPU: std::memcpy", "tab:pink", "dashed"),
        "CPU_MaximumMemoryBandwidth": ("CPU: Theoretical Maximum Memory Bandwidth", "tab:cyan", "dotted"),
        "GPU_NaiveHierarchical": ("GPU: Naive Scan-then-Propagate", "tab:red", "solid"),
        "GPU_OptimizedHierarchical": ("GPU: Optimized Scan-then-Propagate", "tab:orange", "solid"),
        "GPU_OurDecoupledLookback": ("GPU: Our Decoupled Look-Back", "yellow", "solid"),
        "GPU_NvidiaDecoupledLookback": ("GPU: NVIDIA's Decoupled Look-Back", "tab:green", "solid"),
        "GPU_MemoryCopy": ("GPU: cudaMemcpy", "lawngreen", "dashed"),
        "GPU_MaximumMemoryBandwidth": ("GPU: Theoretical Maximum Memory Bandwidth", "lime", "dotted"),
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
) -> float:
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
    if out.returncode:
        warn(f"{out.stderr.strip()}")

    lines = [pattern.match(s) for s in out.stdout.splitlines()]
    times = [float(m.group(1)) for m in lines if m is not None]

    print(f"{scan_type} x {size}: {times}")
    time_avg = mean(times)

    return time_avg

def postprocess_experiment_data(table: dict[str, dict[int, list[int]]]):
    if not all(k in COMMAND_LINE_SCAN_TYPES for k in table.keys()):
        warn(f"unexpected keys: {set(table) - set(COMMAND_LINE_SCAN_TYPES)}")
    print(f"Raw data: {table}")
    avg_table = {
        scan_type: {size: mean(data) for size, data in data_by_size.items()}
        for scan_type, data_by_size in table.items()
    }
    print(f"Plotted data: {avg_table}")
    return avg_table


<<<<<<< HEAD
def plot_timings_with_pandas(df: pd.DataFrame, title_suffix: Optional[str] = None, legend_title: Optional[str] = None):
=======
def plot_timings(df: pd.DataFrame, title_suffix: Optional[str] = None, legend_title: Optional[str] = None):
    # Plotting timings
>>>>>>> 3e538a7 (documentation on performance_test.py)
    plt.figure(figsize=(12, 7.5))
    plt.rcParams["savefig.dpi"] = 240
    plt.rcParams["savefig.transparent"] = True
    sns.set_theme("paper")
    sns.despine()

    plt.grid(axis="y", linestyle="--")
    sns.lineplot(
        data=df,
        x="num_elem",
        y="throughput",
        palette="Set2",
        marker=True,
        style="impl",
        dashes=False,
        markers=["o", "X", "^"],
        markersize=10,
        hue="impl",
    )

    title = "Performance Timing for Inclusive Scan Algorithms"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)
    plt.xlabel("Num. Elements")
    plt.ylabel("Throughput (elem/sec)")
    if legend_title:
        plt.legend(title=legend_title)

    plt.tight_layout()
    plt.savefig("performance-timings.png")


def plot_timings(avg_table: dict[str, dict[int, float]]):
    plt.figure(f"Performance Timing for Inclusive Scan Algorithms")
    plt.title(f"Performance Timing for Inclusive Scan Algorithms")

    # Increase image DPI on save
    plt.rcParams['savefig.dpi'] = 300

    for key, data_by_size in avg_table.items():
        pretty_label, colour, linestyle = get_plot_colour_and_linestyle(key)
        plt.plot(
            data_by_size.keys(),
            data_by_size.values(),
            label=pretty_label,
            color=colour,
            linestyle=linestyle,
        )

    plt.legend()
    plt.xlabel("Input Size")
    plt.ylabel("Time [seconds]")

    plt.savefig(f"performance-timings")


def plot_gpu_timings(avg_table: dict[str, dict[int, float]]):
    plt.figure(f"Performance Timing for Inclusive Scan Algorithms (GPU only)")
    plt.title(f"Performance Timing for Inclusive Scan Algorithms (GPU only)")

    # Increase image DPI on save
    plt.rcParams['savefig.dpi'] = 300

    gpu_avg_table = {
        scan_type: data_by_size
        for scan_type, data_by_size in avg_table.items()
        if scan_type.startswith("GPU_")
    }

    for key, data_by_size in gpu_avg_table.items():
        pretty_label, colour, linestyle = get_plot_colour_and_linestyle(key)
        plt.plot(
            data_by_size.keys(),
            data_by_size.values(),
            label=pretty_label,
            color=colour,
            linestyle=linestyle,
        )

    plt.legend()
    plt.xlabel("Input Size")
    plt.ylabel("Time [seconds]")

    plt.savefig(f"performance-timings")


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
    parser.add_argument("--test-bois", action="store_true", help="Varying bois per thread for cub clone")
    parser.add_argument("--test-boys", action="store_true", help="Varying boys per block for cub clone")
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

<<<<<<< HEAD
    if stevens_implementation:
        sizes = []
        timings = []
        algo = []
        for impl in COMMAND_LINE_SCAN_TYPES:
            for input_size in COMMAND_LINE_INPUT_SIZES:
                t = run_and_time_main(args.executable, impl, input_size, repeats, debug_mode, check_output)
                sizes.append(input_size)
                timings.append(t)
                algo.append(impl)

        table = dict(num_elem=sizes, time=timings, impl=algo)
        df = pd.DataFrame(table)
        df['throughput'] = df.num_elem / df.time
        plot_timings(df, legend_title="Implementation")

    davids_implementation = True
    if davids_implementation:
        table = {
            key: {size: [] for size in COMMAND_LINE_INPUT_SIZES}
            for key in COMMAND_LINE_SCAN_TYPES
        }

        if args.use_cached:
            table = read_result_from_cache("cache.txt")
        else:
            for key in table:
                for size in table[key]:
                    table[key][size] = run_and_time_main(
                        key, size, repeats, debug_mode, check_output
                    )
            write_result_to_cache("cache.txt", table)

        avg_table = postprocess_experiment_data(table)

        plot_timings(avg_table)

        # Plot GPU timing separately if available!
        if not args.cpu_only:
            plot_gpu_timings(avg_table)
=======
    sizes = []
    timings = []
    algo = []
    for impl in COMMAND_LINE_SCAN_TYPES:
        # if impl in ("GPU_CUBSimplified", "GPU_OurDecoupledLookback"):
        #     if args.test_bois:
        #         for i in range(1, 25, 2):
        #             run("rm src/cub_simplified.o".split())
        #             run(f"make BOIS={i}".split())
        #             for input_size in COMMAND_LINE_INPUT_SIZES:
        #                 t = run_and_time_main(args.executable, impl, input_size, repeats, debug_mode, check_output)
        #                 sizes.append(input_size)
        #                 timings.append(t)
        #                 algo.append(f"{impl}_{i}")
        #     elif args.test_boys:
        #         for i in (128, 256, 512):
        #             run("make clean".split())
        #             run(f"make BOYS={i} -j".split())
        #             for input_size in COMMAND_LINE_INPUT_SIZES:
        #                 t = run_and_time_main(args.executable, impl, input_size, repeats, debug_mode, check_output)
        #                 sizes.append(input_size)
        #                 timings.append(t)
        #                 algo.append(f"{impl} / {i}")
        # else:
        for input_size in COMMAND_LINE_INPUT_SIZES:
            t = run_and_time_main(args.executable, impl, input_size, repeats, debug_mode, check_output)
            sizes.append(input_size)
            timings.append(t)
            algo.append(impl)

    table = dict(num_elem=sizes, time=timings, impl=algo)
    df = pd.DataFrame(table)
    df['throughput'] = df.num_elem / df.time
    plot_timings(df, legend_title="Implementation")

    # Plot GPU timing separately if available!
    # if not args.cpu_only:
    #     plot_timings(df[df.impl.str.startswith("GPU")], "GPU only")
>>>>>>> 3e538a7 (documentation on performance_test.py)


if __name__ == "__main__":
    main()

import os
import re
from subprocess import run
from typing import Dict, List
from statistics import mean

import matplotlib.pyplot as plt


COMMAND_LINE_SCAN_TYPES = ["baseline", "decoupled-lookback", "nvidia"]
COMMAND_LINE_INPUT_SIZES = [
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
]


def chdir_to_top_level():
    out = run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        timeout=60,
        text=True,
    )
    path = out.stdout.strip()  # Strip whitespace from path (e.g. trailing \n)
    os.chdir(path)


def run_and_time_main(scan_type: str, size: int) -> List[float]:
    """Time the implementation"""
    assert scan_type in {"baseline", "decoupled-lookback", "nvidia"}
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
            "10",
        ],
        capture_output=True,
        timeout=60,
        text=True,
    )
    lines = [re.match(pattern, s) for s in out.stdout.split("\n")]
    # print(f"Original: {lines}")
    print(f"{out.stdout.splitlines()}")
    times = [float(m.group(1)) for m in lines if m is not None]
    return times


def plot_timings(table: Dict[str, Dict[int, List[int]]]):
    plt.figure("Performance Timing for Inclusive Scan Algorithms")
    plt.title("Performance Timing for Inclusive Scan Algorithms")

    assert all(k in COMMAND_LINE_SCAN_TYPES for k in table.keys())
    avg_table = {
        scan_type: {size: mean(data) for size, data in data_by_size.items()}
        for scan_type, data_by_size in table.items()
    }

    for key, data_by_size in avg_table.items():
        plt.plot(data_by_size.keys(), data_by_size.values(), label=key)

    plt.legend()
    plt.xlabel("Input Size")
    plt.ylabel("Time ")

    plt.savefig("performance-timings")


def main():
    chdir_to_top_level()

    table = {
        key: {size: [] for size in COMMAND_LINE_INPUT_SIZES}
        for key in COMMAND_LINE_SCAN_TYPES
    }

    for key in table:
        for size in table[key]:
            table[key][size] = run_and_time_main(key, size)

    plot_timings(table)


if __name__ == "__main__":
    main()

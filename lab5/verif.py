# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, List
import subprocess
import numpy as np
import csv
import re
import statistics
from collections import defaultdict

EXECUTABLE = Path("./matrix_multi")
SIZES = [200, 400, 800, 1200, 1600, 2000]
PROCS_LIST = [1, 2, 4, 8]
RUNS_PER_CONFIG = 3
CSV_FILE = Path("results.csv")

class BenchmarkEntry:
    def __init__(self, size, procs, time, gflops=None, run_number=1):
        self.size = size
        self.procs = procs
        self.time = time
        self.gflops = gflops
        self.run_number = run_number

def create_matrix(size, path):
    rng = np.random.default_rng(seed=42 + size)
    matrix = rng.uniform(-5, 5, (size, size))
    with path.open("w") as f:
        f.write("{}\n".format(size))
        for row in matrix:
            f.write(" ".join("{:.8f}".format(v) for v in row) + "\n")

def parse_output(output):
    time_val = None
    gflops_val = None
    for line in output.splitlines():
        line_lower = line.lower()
        if "execution time" in line_lower or "time:" in line_lower:
            match = re.search(r"([\d.]+)\s*sec", line)
            if match:
                time_val = float(match.group(1))
        if "gflops" in line_lower:
            match = re.search(r"([\d.]+)\s*gflops", line)
            if match:
                gflops_val = float(match.group(1))
    return time_val, gflops_val

def run_program(a, b, result, num_procs):
    cmd = [
        "srun", "-n", str(num_procs),
        str(EXECUTABLE.resolve()),
        str(a), str(b), str(result)
    ]
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=600
        )
        return process.returncode, process.stdout, process.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout expired"

def execute_case(size, num_procs, run_idx=1):
    file_a = Path("A-{}.txt".format(size))
    file_b = Path("B-{}.txt".format(size))
    file_res = Path("C-{}.txt".format(size))
    
    if not file_a.exists():
        create_matrix(size, file_a)
    if not file_b.exists():
        create_matrix(size, file_b)

    label = "{} proc(s)".format(num_procs)
    print("  [{}/{}] Size={:4d}, {:12s}...".format(run_idx, RUNS_PER_CONFIG, size, label), end=" ")

    code, stdout, stderr = run_program(file_a, file_b, file_res, num_procs)

    if code != 0:
        print("Error: {}".format(stderr.strip()[:100]))
        return BenchmarkEntry(size, num_procs, None, None, run_idx)

    exec_time, gflops = parse_output(stdout)

    if exec_time is None:
        print("No time parsed")
        return BenchmarkEntry(size, num_procs, None, None, run_idx)

    if gflops:
        print("✓ {:.4f}s ({:.2f} GFLOPS)".format(exec_time, gflops))
    else:
        print("✓ {:.4f}s".format(exec_time))

    if file_res.exists():
        file_res.unlink()

    return BenchmarkEntry(size, num_procs, exec_time, gflops, run_idx)

def average_entries(entries):
    groups = defaultdict(list)
    for e in entries:
        if e.time is not None:
            groups[(e.size, e.procs)].append(e.time)
    result = {}
    for (size, procs), times in groups.items():
        result[(size, procs)] = {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times),
            'count': len(times)
        }
    return result

def save_csv(averaged):
    with CSV_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Size", "Processes", "Time_Mean", "Time_Std", "Time_Min", "Time_Max", "Runs"])
        for (size, procs), stats in sorted(averaged.items()):
            writer.writerow([
                size, procs,
                "{:.6f}".format(stats['mean']),
                "{:.6f}".format(stats['std']),
                "{:.6f}".format(stats['min']),
                "{:.6f}".format(stats['max']),
                stats['count']
            ])

def main():
    if not EXECUTABLE.exists():
        raise RuntimeError("Executable not found: {}".format(EXECUTABLE))
    
    print("Running benchmarks: {}".format(EXECUTABLE))
    print("Sizes: {}, Procs: {}, Runs: {}\n".format(SIZES, PROCS_LIST, RUNS_PER_CONFIG))

    all_entries = []
    for size in SIZES:
        print("\nMatrix size: {}x{}".format(size, size))
        for procs in PROCS_LIST:
            for run in range(1, RUNS_PER_CONFIG + 1):
                entry = execute_case(size, procs, run)
                all_entries.append(entry)

    averaged = average_entries(all_entries)
    save_csv(averaged)
    print("\nResults saved: {}".format(CSV_FILE))

    print("\n" + "="*70)
    print("SUMMARY RESULTS (mean time, sec):")
    print("="*70)
    print("{:>6} | ".format("Size"), end="")
    for p in PROCS_LIST:
        print(" {:>6} proc | ".format(p), end="")
    print()
    print("-"*70)
    for size in SIZES:
        print("{:6d} | ".format(size), end="")
        for procs in PROCS_LIST:
            stats = averaged.get((size, procs))
            if stats:
                print(" {:8.4f} | ".format(stats['mean']), end="")
            else:
                print(" {:>8} | ".format("N/A"), end="")
        print()

if __name__ == "__main__":
    main()
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import subprocess
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import statistics
from collections import defaultdict

EXECUTABLE = Path("./lab3")
SIZES = [200, 400, 800, 1200, 1600, 2000]
PROCS_LIST = [1, 2, 4, 6, 8, 10]
RUNS_PER_CONFIG = 3

CSV_FILE = Path("results.csv")
PLOT_TIME_FILE = Path("plot_time.png")
PLOT_SPEEDUP_FILE = Path("plot_speedup.png")
PLOT_EFFICIENCY_FILE = Path("plot_efficiency.png")


@dataclass
class BenchmarkEntry:
    size: int
    procs: int
    time: Optional[float]
    gflops: Optional[float] = None
    run_number: int = 1


def create_matrix(size: int, path: Path) -> None:
    rng = np.random.default_rng(seed=42 + size)
    matrix = rng.uniform(-5, 5, (size, size))

    with path.open("w") as f:
        f.write(f"{size}\n")
        for row in matrix:
            f.write(" ".join(f"{v:.8f}" for v in row) + "\n")


def parse_output(output: str) -> tuple[Optional[float], Optional[float]]:
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


def run_program(a: Path, b: Path, result: Path, num_procs: int) -> tuple[int, str, str]:
    cmd = [
        "mpirun", 
        "-np", str(num_procs),
        str(EXECUTABLE.resolve()), 
        str(a), str(b), str(result)
    ]
    
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return process.returncode, process.stdout, process.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout expired"


def execute_case(size: int, num_procs: int, run_idx: int = 1) -> BenchmarkEntry:
    file_a = Path(f"A-{size}.txt")
    file_b = Path(f"B-{size}.txt")
    file_res = Path(f"C-{size}.txt")

    if not file_a.exists():
        create_matrix(size, file_a)
    if not file_b.exists():
        create_matrix(size, file_b)

    label = f"{num_procs} proc(s)"
    print(f"  [{run_idx}/{RUNS_PER_CONFIG}] Size={size:4d}, {label:12s}...", end=" ")

    code, stdout, stderr = run_program(file_a, file_b, file_res, num_procs)

    if code != 0:
        print(f"Error: {stderr.strip()[:100]}")
        return BenchmarkEntry(size, num_procs, None, None, run_idx)

    exec_time, gflops = parse_output(stdout)

    if exec_time is None:
        print("No time parsed")
        return BenchmarkEntry(size, num_procs, None, None, run_idx)

    print(f"✓ {exec_time:.4f}s ({gflops:.2f} GFLOPS)" if gflops else f" {exec_time:.4f}s")
    
    if file_res.exists():
        file_res.unlink()

    return BenchmarkEntry(size, num_procs, exec_time, gflops, run_idx)


def average_entries(entries: List[BenchmarkEntry]) -> dict[tuple[int, int], dict]:
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


def save_csv(averaged: dict[tuple[int, int], dict]) -> None:
    with CSV_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Size", "Processes", "Time_Mean", "Time_Std", "Time_Min", "Time_Max", "Runs"])
        
        for (size, procs), stats in sorted(averaged.items()):
            writer.writerow([
                size,
                procs,
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
                stats['count']
            ])


def plot_time(averaged: dict[tuple[int, int], dict]) -> None:
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(PROCS_LIST)))
    
    for idx, procs in enumerate(PROCS_LIST):
        sizes = []
        times = []
        errors = []
        
        for (size, p), stats in averaged.items():
            if p == procs:
                sizes.append(size)
                times.append(stats['mean'])
                errors.append(stats['std'])
        
        if sizes:
            plt.errorbar(sizes, times, yerr=errors, marker='o', label=f'{procs} process(es)', 
                        color=colors[idx], capsize=3, alpha=0.8)
    
    plt.xlabel("Размер матрицы (N×N)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("MPI: Умножение матриц - время от размера")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_TIME_FILE, dpi=150)
    print(f"График времени сохранён: {PLOT_TIME_FILE}")


def plot_speedup(averaged: dict[tuple[int, int], dict]) -> None:
    plt.figure(figsize=(10, 6))
    
    for size in SIZES:
        base_time = averaged.get((size, 1), {}).get('mean')
        if base_time is None or base_time == 0:
            continue
            
        procs_list = []
        speedups = []
        
        for procs in PROCS_LIST:
            t = averaged.get((size, procs), {}).get('mean')
            if t and t > 0:
                procs_list.append(procs)
                speedups.append(base_time / t)
        
        if speedups:
            plt.plot(procs_list, speedups, marker='o', label=f'N={size}', alpha=0.8)
    
    max_p = max(PROCS_LIST)
    plt.plot(PROCS_LIST, PROCS_LIST, 'k--', label='Ideal speedup', alpha=0.4)
    
    plt.xlabel("Количество процессов")
    plt.ylabel("Ускорение (Speedup)")
    plt.title("MPI: Эффективность параллелизации (относительно 1 процесса)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_SPEEDUP_FILE, dpi=150)
    print(f"График ускорения сохранён: {PLOT_SPEEDUP_FILE}")


def plot_efficiency(averaged: dict[tuple[int, int], dict]) -> None:
    plt.figure(figsize=(10, 6))
    
    for size in SIZES:
        base_time = averaged.get((size, 1), {}).get('mean')
        if base_time is None or base_time == 0:
            continue
            
        procs_list = []
        efficiencies = []
        
        for procs in PROCS_LIST:
            if procs == 1:
                continue
            t = averaged.get((size, procs), {}).get('mean')
            if t and t > 0:
                speedup = base_time / t
                efficiency = (speedup / procs) * 100
                procs_list.append(procs)
                efficiencies.append(efficiency)
        
        if efficiencies:
            plt.plot(procs_list, efficiencies, marker='s', label=f'N={size}', alpha=0.8)
    
    plt.axhline(y=100, color='k', linestyle='--', label='Ideal (100%)', alpha=0.4)
    
    plt.xlabel("Количество процессов")
    plt.ylabel("Эффективность (%)")
    plt.title("MPI: Эффективность использования процессов")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_EFFICIENCY_FILE, dpi=150)
    print(f"График эффективности сохранён: {PLOT_EFFICIENCY_FILE}")


def main() -> None:
    if not EXECUTABLE.exists():
        raise RuntimeError(f"Исполняемый файл не найден: {EXECUTABLE}")
    try:
        subprocess.run(["mpirun", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("mpirun не найден.")

    print(f"Запуск MPI бенчмарков: {EXECUTABLE}")
    print(f"Размеры: {SIZES}")
    print(f"Процессы: {PROCS_LIST}")
    print(f"Повторов: {RUNS_PER_CONFIG}\n")

    all_entries = []

    for size in SIZES:
        print(f"\nРазмер матрицы: {size}×{size}")
        
        for procs in PROCS_LIST:
            for run in range(1, RUNS_PER_CONFIG + 1):
                entry = execute_case(size, procs, run)
                all_entries.append(entry)

    averaged = average_entries(all_entries)
    save_csv(averaged)
    print(f"\nРезультаты сохранены: {CSV_FILE}")
    
    plot_time(averaged)
    plot_speedup(averaged)
    plot_efficiency(averaged)
    
    print("\n" + "="*70)
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ MPI (среднее время, сек):")
    print("="*70)
    print(f"{'Size':>6} |", end="")
    for p in PROCS_LIST:
        print(f" {p:>10} proc |", end="")
    print()
    print("-"*70)
    
    for size in SIZES:
        print(f"{size:6d} |", end="")
        for procs in PROCS_LIST:
            stats = averaged.get((size, procs))
            if stats:
                print(f" {stats['mean']:10.4f} |", end="")
            else:
                print(f" {'N/A':>10} |", end="")
        print()


if __name__ == "__main__":
    main()
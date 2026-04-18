from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import subprocess
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import statistics
from collections import defaultdict

EXECUTABLE = Path("./lab4.exe")
SIZES = [200, 400, 800, 1200, 1600, 2000]
BLOCK_SIZES = [8, 16, 32]
RUNS_PER_CONFIG = 3

CSV_FILE = Path("results.csv")
PLOT_TIME_FILE = Path("plot_time.png")
PLOT_PERF_FILE = Path("plot_performance.png")


@dataclass
class BenchmarkEntry:
    size: int
    block_size: int
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
        if "execution time" in line_lower:
            match = re.search(r"([\d.]+)\s*sec", line)
            if match:
                time_val = float(match.group(1))
        if "performance" in line_lower or "gflops" in line_lower:
            match = re.search(r"([\d.]+)\s*gflops", line)
            if match:
                gflops_val = float(match.group(1))
    
    return time_val, gflops_val


def run_program(a: Path, b: Path, result: Path, block_size: int, use_shared: int = 0) -> tuple[int, str, str]:
    cmd = [
        str(EXECUTABLE.resolve()), 
        str(a), str(b), str(result),
        str(block_size),
        str(use_shared)
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


def execute_case(size: int, block_size: int, run_idx: int = 1, use_shared: int = 0) -> BenchmarkEntry:
    file_a = Path(f"A-{size}.txt")
    file_b = Path(f"B-{size}.txt")
    file_res = Path(f"C-{size}.txt")

    if not file_a.exists():
        create_matrix(size, file_a)
    if not file_b.exists():
        create_matrix(size, file_b)

    label = f"block={block_size}" + (" (shared)" if use_shared else "")
    print(f"  [{run_idx}/{RUNS_PER_CONFIG}] Size={size:4d}, {label:20s}...", end=" ")

    code, stdout, stderr = run_program(file_a, file_b, file_res, block_size, use_shared)

    if code != 0:
        print(f"Error: {stderr.strip()[:100]}")
        return BenchmarkEntry(size, block_size, None, None, run_idx)

    exec_time, gflops = parse_output(stdout)

    if exec_time is None:
        print("No time parsed")
        return BenchmarkEntry(size, block_size, None, None, run_idx)

    print(f"✓ {exec_time:.6f}s ({gflops:.2f} GFLOPS)" if gflops else f" {exec_time:.6f}s")
    
    if file_res.exists():
        file_res.unlink()

    return BenchmarkEntry(size, block_size, exec_time, gflops, run_idx)


def average_entries(entries: List[BenchmarkEntry]) -> dict[tuple[int, int], dict]:
    groups = defaultdict(list)
    for e in entries:
        if e.time is not None:
            groups[(e.size, e.block_size)].append(e.time)
    
    result = {}
    for (size, block_size), times in groups.items():
        result[(size, block_size)] = {
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
        writer.writerow(["Size", "Block_Size", "Time_Mean", "Time_Std", "Time_Min", "Time_Max", "Runs"])
        
        for (size, block_size), stats in sorted(averaged.items()):
            writer.writerow([
                size,
                block_size,
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
                stats['count']
            ])


def plot_time(averaged: dict[tuple[int, int], dict]) -> None:
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(BLOCK_SIZES)))
    
    for idx, block_size in enumerate(BLOCK_SIZES):
        sizes = []
        times = []
        errors = []
        
        for (size, bs), stats in averaged.items():
            if bs == block_size:
                sizes.append(size)
                times.append(stats['mean'])
                errors.append(stats['std'])
        
        if sizes:
            plt.errorbar(sizes, times, yerr=errors, marker='o', label=f'{block_size}×{block_size} blocks', 
                        color=colors[idx], capsize=3, alpha=0.8)
    
    plt.xlabel("Размер матрицы (N×N)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("CUDA: Умножение матриц - время от размера")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_TIME_FILE, dpi=150)
    print(f"График времени сохранён: {PLOT_TIME_FILE}")


def plot_performance(averaged: dict[tuple[int, int], dict]) -> None:
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(BLOCK_SIZES)))
    
    for idx, block_size in enumerate(BLOCK_SIZES):
        sizes = []
        gflops = []
        
        for (size, bs), stats in averaged.items():
            if bs == block_size and stats['mean'] > 0:
                sizes.append(size)
                ops = 2 * size * size * size
                gflops_val = ops / stats['mean'] / 1e9
                gflops.append(gflops_val)
        
        if gflops:
            plt.plot(sizes, gflops, marker='s', label=f'{block_size}×{block_size} blocks', 
                    color=colors[idx], alpha=0.8)
    
    plt.xlabel("Размер матрицы (N×N)")
    plt.ylabel("Производительность (GFLOPS)")
    plt.title("CUDA: Производительность умножения матриц")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PERF_FILE, dpi=150)
    print(f"График производительности сохранён: {PLOT_PERF_FILE}")


def main() -> None:
    if not EXECUTABLE.exists():
        raise RuntimeError(f"Исполняемый файл не найден: {EXECUTABLE}")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("NVIDIA GPU не найден или CUDA не установлена")

    print(f"Запуск CUDA бенчмарков: {EXECUTABLE}")
    print(f"Размеры: {SIZES}")
    print(f"Размеры блоков: {BLOCK_SIZES}")
    print(f"Повторов: {RUNS_PER_CONFIG}\n")

    all_entries = []

    for size in SIZES:
        print(f"\nРазмер матрицы: {size}×{size}")
        
        for block_size in BLOCK_SIZES:
            for run in range(1, RUNS_PER_CONFIG + 1):
                entry = execute_case(size, block_size, run)
                all_entries.append(entry)

    averaged = average_entries(all_entries)
    save_csv(averaged)
    print(f"\nРезультаты сохранены: {CSV_FILE}")
    
    plot_time(averaged)
    plot_performance(averaged)
    
    print("\n" + "="*70)
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ CUDA (среднее время, сек):")
    print("="*70)
    print(f"{'Size':>6} |", end="")
    for bs in BLOCK_SIZES:
        print(f" {bs:>10} block |", end="")
    print()
    print("-"*70)
    
    for size in SIZES:
        print(f"{size:6d} |", end="")
        for block_size in BLOCK_SIZES:
            stats = averaged.get((size, block_size))
            if stats:
                print(f" {stats['mean']:10.6f} |", end="")
            else:
                print(f" {'N/A':>10} |", end="")
        print()
    

if __name__ == "__main__":
    main()
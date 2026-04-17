from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import subprocess
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import statistics

EXECUTABLE = Path("./lab2")
SIZES = [200, 400, 800, 1200, 1600, 2000]
THREADS_LIST = [1, 2, 4, 8, 10]
RUNS_PER_CONFIG = 3

CSV_FILE = Path("results.csv")
PLOT_TIME_FILE = Path("plot_time.png")
PLOT_SPEEDUP_FILE = Path("plot_speedup.png")


@dataclass
class BenchmarkEntry:
    size: int
    threads: int
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


def run_program(a: Path, b: Path, result: Path,
                threads: Optional[int] = None,
                sequential: bool = False) -> tuple[int, str, str]:
    cmd = [str(EXECUTABLE.resolve()), str(a), str(b), str(result)]
    
    if sequential:
        cmd.append("seq")
    elif threads is not None:
        cmd.append(str(threads))
    
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    return process.returncode, process.stdout, process.stderr


def execute_case(size: int, threads: int, run_idx: int = 1) -> BenchmarkEntry:
    file_a = Path(f"A-{size}.txt")
    file_b = Path(f"B-{size}.txt")
    file_res = Path(f"C-{size}.txt")

    if not file_a.exists():
        create_matrix(size, file_a)
    if not file_b.exists():
        create_matrix(size, file_b)

    sequential = (threads == 0)
    label = "seq" if sequential else f"{threads} threads"
    print(f"  [{run_idx}/{RUNS_PER_CONFIG}] Size={size:4d}, {label:10s}...", end=" ")

    code, stdout, stderr = run_program(file_a, file_b, file_res,
                                       threads=None if sequential else threads,
                                       sequential=sequential)

    if code != 0:
        print(f"Error: {stderr.strip()[:100]}")
        return BenchmarkEntry(size, threads, None, None, run_idx)

    exec_time, gflops = parse_output(stdout)

    if exec_time is None:
        print("No time parsed")
        return BenchmarkEntry(size, threads, None, None, run_idx)

    print(f"✓ {exec_time:.4f}s ({gflops:.2f} GFLOPS)" if gflops else f"✓ {exec_time:.4f}s")
    
    if file_res.exists():
        file_res.unlink()

    return BenchmarkEntry(size, threads, exec_time, gflops, run_idx)


def average_entries(entries: List[BenchmarkEntry]) -> dict[tuple[int, int], dict]:
    from collections import defaultdict
    
    groups = defaultdict(list)
    for e in entries:
        if e.time is not None:
            groups[(e.size, e.threads)].append(e.time)
    
    result = {}
    for (size, threads), times in groups.items():
        result[(size, threads)] = {
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
        writer.writerow(["Size", "Threads", "Time_Mean", "Time_Std", "Time_Min", "Time_Max", "Runs"])
        
        for (size, threads), stats in sorted(averaged.items()):
            writer.writerow([
                size,
                threads,
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
                stats['count']
            ])


def plot_time(averaged: dict[tuple[int, int], dict]) -> None:
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(THREADS_LIST)))
    
    for idx, threads in enumerate(THREADS_LIST):
        sizes = []
        times = []
        errors = []
        
        for (size, t), stats in averaged.items():
            if t == threads:
                sizes.append(size)
                times.append(stats['mean'])
                errors.append(stats['std'])
        
        if sizes:
            plt.errorbar(sizes, times, yerr=errors, marker='o', label=f'{threads} thread(s)', 
                        color=colors[idx], capsize=3, alpha=0.8)
    
    plt.xlabel("Размер матрицы (N×N)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Умножение матриц: время от размера")
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
            
        sizes = []
        speedups = []
        
        for threads in THREADS_LIST:
            t = averaged.get((size, threads), {}).get('mean')
            if t and t > 0:
                sizes.append(threads)
                speedups.append(base_time / t)
        
        if speedups:
            plt.plot(sizes, speedups, marker='o', label=f'N={size}', alpha=0.8)
    
    max_t = max(THREADS_LIST)
    plt.plot(THREADS_LIST, THREADS_LIST, 'k--', label='Ideal speedup', alpha=0.4)
    
    plt.xlabel("Количество потоков")
    plt.ylabel("Ускорение (Speedup)")
    plt.title("Эффективность параллелизации (относительно 1 потока)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_SPEEDUP_FILE, dpi=150)
    print(f"График ускорения сохранён: {PLOT_SPEEDUP_FILE}")


def main() -> None:
    if not EXECUTABLE.exists():
        raise RuntimeError(f"Исполняемый файл не найден: {EXECUTABLE}")

    print(f"Запуск бенчмарков: {EXECUTABLE}")
    print(f"Размеры: {SIZES}")
    print(f"Потоки: {THREADS_LIST}")
    print(f"Повторов: {RUNS_PER_CONFIG}\n")

    all_entries = []

    for size in SIZES:
        print(f"\nРазмер матрицы: {size}×{size}")
        
        for threads in THREADS_LIST:
            for run in range(1, RUNS_PER_CONFIG + 1):
                entry = execute_case(size, threads, run)
                all_entries.append(entry)

    averaged = average_entries(all_entries)
    save_csv(averaged)
    print(f"\n💾 Результаты сохранены: {CSV_FILE}")
    
    plot_time(averaged)
    plot_speedup(averaged)
    
    print("\n" + "="*60)
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ (среднее время, сек):")
    print("="*60)
    print(f"{'Size':>6} |", end="")
    for t in THREADS_LIST:
        print(f" {t:>8} thr |", end="")
    print()
    print("-"*60)
    
    for size in SIZES:
        print(f"{size:6d} |", end="")
        for threads in THREADS_LIST:
            stats = averaged.get((size, threads))
            if stats:
                print(f" {stats['mean']:8.4f} |", end="")
            else:
                print(f" {'N/A':>8} |", end="")
        print()


if __name__ == "__main__":
    main()
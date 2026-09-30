import argparse
import asyncio
import math
import os
import random
import time

from multiprocessing import Pool


def monte_carlo_serial(samples):
    hits = 0

    for _ in range(samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x * x + y * y <= 1:
            hits += 1

    return hits


def monte_carlo_partial(samples):
    hits = 0

    for _ in range(samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x * x + y * y <= 1:
            hits += 1

    return hits


async def monitor_task(task_id, log_file, duration):
    start = time.time()

    while time.time() - start < duration:
        with open(log_file, "a") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"| tarefa={task_id} "
                f"| pid={os.getpid()} "
                f"| monitoramento ativo\n"
            )

        await asyncio.sleep(0.5)


async def run_async_monitors(log_file, duration):
    tasks = [
        asyncio.create_task(
            monitor_task(i, log_file, duration)
        )
        for i in range(20)
    ]

    await asyncio.gather(*tasks)


def write_log(log_file, message):
    with open(log_file, "a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"| pid={os.getpid()} "
            f"| {message}\n"
        )


def run_serial(samples, log_file):
    start = time.perf_counter()

    hits = monte_carlo_serial(samples)

    elapsed = time.perf_counter() - start
    pi = 4 * hits / samples

    write_log(
        log_file,
        f"serial concluido | samples={samples} "
        f"| pi={pi} | tempo={elapsed:.6f}"
    )

    return pi, elapsed


def run_parallel(samples, workers, n_tasks, chunk_size, log_file):
    start = time.perf_counter()

    total_samples = n_tasks * chunk_size

    if samples is not None:
        total_samples = samples

        n_tasks = math.ceil(
            samples / chunk_size
        )

    tasks = []

    remaining = total_samples

    for _ in range(n_tasks):
        current_chunk = min(
            chunk_size,
            remaining
        )

        if current_chunk <= 0:
            break

        tasks.append(current_chunk)
        remaining -= current_chunk

    write_log(
        log_file,
        f"inicio paralelo | workers={workers} "
        f"| n_tasks={n_tasks} "
        f"| chunk_size={chunk_size} "
        f"| total={total_samples}"
    )

    async_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(async_loop)

    monitor = async_loop.create_task(
        run_async_monitors(
            log_file,
            max(1, 60)
        )
    )

    with Pool(processes=workers) as pool:
        results = []

        for index, result in enumerate(
            pool.imap_unordered(
                monte_carlo_partial,
                tasks
            ),
            start=1
        ):
            results.append(result)

            completed = sum(
                tasks[:index]
            )

            write_log(
                log_file,
                f"progresso={index}/{len(tasks)} "
                f"| samples_processados={completed}"
            )

            # Permite que as tarefas asyncio executem
            async_loop.run_until_complete(
                asyncio.sleep(0)
            )

    monitor.cancel()

    try:
        async_loop.run_until_complete(monitor)
    except asyncio.CancelledError:
        pass

    async_loop.close()

    hits = sum(results)

    elapsed = time.perf_counter() - start
    pi = 4 * hits / total_samples

    write_log(
        log_file,
        f"paralelo concluido | samples={total_samples} "
        f"| pi={pi} | tempo={elapsed:.6f}"
    )

    return pi, elapsed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["serial", "parallel"],
        required=True
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--n-tasks",
        type=int,
        default=20
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000
    )

    parser.add_argument(
        "--log",
        default="monte_carlo.log"
    )

    args = parser.parse_args()

    if args.mode == "serial":

        if args.samples is None:
            raise ValueError(
                "--samples é obrigatório no modo serial"
            )

        pi, elapsed = run_serial(
            args.samples,
            args.log
        )

    else:

        pi, elapsed = run_parallel(
            args.samples,
            args.workers,
            args.n_tasks,
            args.chunk_size,
            args.log
        )

    print()
    print("===== MONTE CARLO =====")
    print(f"Modo: {args.mode}")
    print(f"Pi estimado: {pi:.10f}")
    print(f"Tempo: {elapsed:.6f} segundos")
    print(f"Log: {args.log}")


if __name__ == "__main__":
    main()
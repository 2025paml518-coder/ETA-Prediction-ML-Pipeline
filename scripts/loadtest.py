"""Latency and throughput benchmark for the prediction service (M4).

Reports percentiles rather than an average. The mean latency of a service is the
one number that describes nobody's experience: it hides the tail, and the tail is
what a caller with a timeout actually hits.

Two things are measured separately:

* **Single-request latency** at a given concurrency, which is what an end user waits.
* **Batch throughput**, which is what a bulk re-scoring job cares about. Batching
  amortises featurisation and makes one vectorised model call, so per-trip cost
  falls sharply with size - the gap between the two is the argument for the
  /predict/batch endpoint existing at all.

Usage:
    python -m scripts.loadtest --url http://localhost:8000 --requests 500 --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

from src.config import project_path
from src.utils.io import atomic_write_json, atomic_write_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

HOTSPOTS = [
    (40.7549, -73.9840),
    (40.7075, -74.0113),
    (40.6413, -73.7781),
    (40.7769, -73.8740),
    (40.6928, -73.9903),
    (40.8116, -73.9465),
]
WEATHER = ["Clear", "Cloudy", "Rain", "Snow", "Fog"]


def sample_trip(rng: random.Random) -> dict:
    pickup = rng.choice(HOTSPOTS)
    dropoff = rng.choice(HOTSPOTS)
    condition = rng.choice(WEATHER)
    return {
        "pickup_datetime": f"2024-07-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:30:00",
        "pickup_latitude": round(pickup[0] + rng.uniform(-0.01, 0.01), 6),
        "pickup_longitude": round(pickup[1] + rng.uniform(-0.01, 0.01), 6),
        "dropoff_latitude": round(dropoff[0] + rng.uniform(-0.01, 0.01), 6),
        "dropoff_longitude": round(dropoff[1] + rng.uniform(-0.01, 0.01), 6),
        "traffic_index": round(rng.uniform(0.05, 0.95), 4),
        "vendor_id": rng.choice([1, 2, 3]),
        "passenger_count": rng.randint(1, 6),
        "store_and_fwd_flag": "N",
        "weather_condition": condition,
        "temperature_c": round(rng.uniform(-5, 35), 1),
        "precipitation_mm": round(rng.uniform(0.1, 20), 2) if condition in {"Rain", "Snow"} else 0.0,
        "wind_kph": round(rng.uniform(0, 40), 1),
    }


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(int(fraction * len(ordered)), len(ordered) - 1)
        return ordered[index]

    return {
        "min": round(ordered[0], 3),
        "p50": round(at(0.50), 3),
        "p90": round(at(0.90), 3),
        "p95": round(at(0.95), 3),
        "p99": round(at(0.99), 3),
        "max": round(ordered[-1], 3),
        "mean": round(statistics.fmean(ordered), 3),
    }


async def _worker(
    client: httpx.AsyncClient, url: str, payloads: list[dict], latencies: list[float], errors: list[int]
) -> None:
    for payload in payloads:
        started = time.perf_counter()
        try:
            response = await client.post(f"{url}/predict", json=payload, timeout=30.0)
            latencies.append((time.perf_counter() - started) * 1000.0)
            if response.status_code != 200:
                errors.append(response.status_code)
        except httpx.HTTPError:
            errors.append(0)


async def measure_single(url: str, total: int, concurrency: int, rng: random.Random) -> dict:
    payloads = [sample_trip(rng) for _ in range(total)]
    chunks = [payloads[i::concurrency] for i in range(concurrency)]
    latencies: list[float] = []
    errors: list[int] = []

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        started = time.perf_counter()
        await asyncio.gather(
            *(_worker(client, url, chunk, latencies, errors) for chunk in chunks)
        )
        wall = time.perf_counter() - started

    return {
        "requests": total,
        "concurrency": concurrency,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(total / wall, 2) if wall else 0.0,
        "errors": len(errors),
        "latency_ms": percentiles(latencies) if latencies else {},
    }


async def measure_batch(url: str, sizes: list[int], rng: random.Random) -> list[dict]:
    results = []
    async with httpx.AsyncClient() as client:
        for size in sizes:
            payload = {"trips": [sample_trip(rng) for _ in range(size)]}
            started = time.perf_counter()
            response = await client.post(f"{url}/predict/batch", json=payload, timeout=60.0)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            results.append(
                {
                    "batch_size": size,
                    "status": response.status_code,
                    "total_ms": round(elapsed_ms, 3),
                    "ms_per_trip": round(elapsed_ms / size, 4),
                    "trips_per_second": round(size / (elapsed_ms / 1000.0), 1) if elapsed_ms else 0,
                }
            )
    return results


def render(report: dict) -> str:
    single = report["single"]
    latency = single["latency_ms"]

    lines = [
        "# API Latency and Throughput",
        "",
        f"Target: `{report['url']}` | model `{report['model_version']}` | "
        f"generated {report['generated_at']}",
        "",
        "## Single-trip requests",
        "",
        f"{single['requests']} requests at concurrency {single['concurrency']}, "
        f"{single['errors']} errors.",
        "",
        "| Metric | ms |",
        "| --- | --- |",
        f"| min | {latency['min']} |",
        f"| p50 | {latency['p50']} |",
        f"| p90 | {latency['p90']} |",
        f"| p95 | {latency['p95']} |",
        f"| p99 | {latency['p99']} |",
        f"| max | {latency['max']} |",
        f"| mean | {latency['mean']} |",
        "",
        f"Throughput: **{single['throughput_rps']} req/s** sustained over "
        f"{single['wall_seconds']}s.",
        "",
        "## Batch requests",
        "",
        "| Batch size | Total ms | ms per trip | Trips/s |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["batch"]:
        lines.append(
            f"| {row['batch_size']} | {row['total_ms']} | {row['ms_per_trip']} | "
            f"{row['trips_per_second']} |"
        )

    smallest = report["batch"][0]
    largest = report["batch"][-1]
    speedup = smallest["ms_per_trip"] / largest["ms_per_trip"] if largest["ms_per_trip"] else 0

    lines += [
        "",
        "## Reading these numbers",
        "",
        f"- p99 is {latency['p99'] / max(latency['p50'], 1e-9):.1f}x p50. The gap is the "
        "queueing tail under concurrency, which an average would hide entirely.",
        f"- Batching {largest['batch_size']} trips costs {largest['ms_per_trip']}ms per trip "
        f"against {smallest['ms_per_trip']}ms at size {smallest['batch_size']}, a "
        f"{speedup:.1f}x reduction. Featurisation and the model call are both vectorised, "
        "so the fixed per-request overhead is amortised across the payload.",
        "- Single-worker figures. Uvicorn workers each hold their own copy of the model, so "
        "horizontal scaling trades memory for concurrency.",
        "",
    ]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)

    async with httpx.AsyncClient() as client:
        ready = await client.get(f"{args.url}/ready", timeout=30.0)
        if ready.status_code != 200:
            raise SystemExit(f"Service not ready at {args.url}: {ready.text}")
        info = (await client.get(f"{args.url}/model/info", timeout=30.0)).json()

    logger.info("Warming up with %s requests", args.warmup)
    await measure_single(args.url, args.warmup, min(args.concurrency, 4), rng)

    logger.info("Measuring %s requests at concurrency %s", args.requests, args.concurrency)
    single = await measure_single(args.url, args.requests, args.concurrency, rng)

    logger.info("Measuring batch throughput")
    batch = await measure_batch(args.url, [1, 10, 50, 100, 250], rng)

    return {
        "url": args.url,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_version": info.get("model_version", "unknown"),
        "single": single,
        "batch": batch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the ETA prediction API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="reports/api")
    args = parser.parse_args()

    report = asyncio.run(run(args))

    destination = Path(project_path(args.output_dir))
    atomic_write_json(report, destination / "latency_report.json")
    atomic_write_text(render(report), destination / "latency_report.md")

    print(json.dumps(report["single"], indent=2))
    logger.info("Wrote %s", destination / "latency_report.md")


if __name__ == "__main__":
    main()

# API Latency and Throughput

Target: `http://127.0.0.1:8000` | model `3` | generated 2026-08-16T12:07:50

## Single-trip requests

500 requests at concurrency 16, 0 errors.

| Metric | ms |
| --- | --- |
| min | 61.64 |
| p50 | 332.277 |
| p90 | 450.58 |
| p95 | 515.927 |
| p99 | 742.081 |
| max | 1103.859 |
| mean | 338.916 |

Throughput: **45.45 req/s** sustained over 11.001s.

## Batch requests

| Batch size | Total ms | ms per trip | Trips/s |
| --- | --- | --- | --- |
| 1 | 29.324 | 29.3244 | 34.1 |
| 10 | 25.75 | 2.575 | 388.3 |
| 50 | 43.933 | 0.8787 | 1138.1 |
| 100 | 64.194 | 0.6419 | 1557.8 |
| 250 | 98.013 | 0.3921 | 2550.7 |

## Reading these numbers

- p99 is 2.2x p50. The gap is the queueing tail under concurrency, which an average would hide entirely.
- Batching 250 trips costs 0.3921ms per trip against 29.3244ms at size 1, a 74.8x reduction. Featurisation and the model call are both vectorised, so the fixed per-request overhead is amortised across the payload.
- Single-worker figures. Uvicorn workers each hold their own copy of the model, so horizontal scaling trades memory for concurrency.

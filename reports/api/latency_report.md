# API Latency and Throughput

Target: `http://127.0.0.1:8000` | model `2` | generated 2026-08-21T13:49:28

## Single-trip requests

500 requests at concurrency 16, 0 errors.

| Metric | ms |
| --- | --- |
| min | 116.054 |
| p50 | 218.79 |
| p90 | 255.041 |
| p95 | 299.656 |
| p99 | 363.947 |
| max | 429.67 |
| mean | 221.988 |

Throughput: **71.39 req/s** sustained over 7.003s.

## Batch requests

| Batch size | Total ms | ms per trip | Trips/s |
| --- | --- | --- | --- |
| 1 | 19.594 | 19.5939 | 51.0 |
| 10 | 21.889 | 2.1889 | 456.8 |
| 50 | 32.546 | 0.6509 | 1536.3 |
| 100 | 58.795 | 0.5879 | 1700.8 |
| 250 | 78.308 | 0.3132 | 3192.5 |

## Reading these numbers

- p99 is 1.7x p50. The gap is the queueing tail under concurrency, which an average would hide entirely.
- Batching 250 trips costs 0.3132ms per trip against 19.5939ms at size 1, a 62.6x reduction. Featurisation and the model call are both vectorised, so the fixed per-request overhead is amortised across the payload.
- Single-worker figures. Uvicorn workers each hold their own copy of the model, so horizontal scaling trades memory for concurrency.

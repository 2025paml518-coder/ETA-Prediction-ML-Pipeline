# Level 3 Baseline Profile

Built from 101,471 training rows | reference sample per column: 2,000

## Continuous features

| Feature | Mean | Std | p1 | p50 | p99 | Null rate |
| --- | --- | --- | --- | --- | --- | --- |
| `trip_duration_min` | 24.4486 | 16.6827 | 3.4916 | 19.9014 | 80.1728 | 0.0000% |
| `avg_speed_kmph` | 20.6266 | 6.6235 | 7.3166 | 20.0211 | 38.5636 | 0.0000% |
| `traffic_index` | 0.4714 | 0.2246 | 0.0200 | 0.4970 | 0.8933 | 0.0000% |
| `temperature_c` | 14.2815 | 10.3712 | -5.4000 | 15.9000 | 32.8000 | 0.7992% |
| `precipitation_mm` | 0.8241 | 2.2421 | 0.0000 | 0.0000 | 10.3300 | 0.7992% |
| `wind_kph` | 13.1150 | 6.9697 | 3.2000 | 12.1000 | 34.7000 | 0.7992% |
| `passenger_count` | 1.7864 | 1.3020 | 1.0000 | 1.0000 | 6.0000 | 0.3981% |
| `pickup_latitude` | 40.7404 | 0.0501 | 40.6352 | 40.7503 | 40.8409 | 0.0000% |
| `pickup_longitude` | -73.9554 | 0.0558 | -74.0264 | -73.9674 | -73.7731 | 0.0000% |
| `dropoff_latitude` | 40.7406 | 0.0502 | 40.6351 | 40.7504 | 40.8408 | 0.0000% |
| `dropoff_longitude` | -73.9552 | 0.0557 | -74.0260 | -73.9672 | -73.7731 | 0.0000% |

## Categorical features

| Feature | Category | Share |
| --- | --- | --- |
| `weather_condition` | Clear | 45.7709% |
| `weather_condition` | Cloudy | 29.9493% |
| `weather_condition` | Rain | 20.2742% |
| `weather_condition` | Snow | 3.0936% |
| `weather_condition` | Fog | 0.9120% |
| `vendor_id` | 1 | 44.9754% |
| `vendor_id` | 2 | 40.0489% |
| `vendor_id` | 3 | 14.9757% |
| `store_and_fwd_flag` | N | 98.5286% |
| `store_and_fwd_flag` | Y | 1.4714% |

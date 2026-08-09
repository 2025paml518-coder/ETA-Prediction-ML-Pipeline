"""Synthetic NYC trip generator with a documented data-generating process.

Why synthetic rather than the raw NYC Taxi Trip Duration dump:

1. The brief requires *weather* and *traffic* features. The public taxi dataset has
   neither, so they would have to be fabricated and joined in anyway.
2. Drift simulation (Week 4 / M5) needs a controllable generating process. Owning the
   process is what makes "festival surge" or "monsoon" a *ground truth* shift rather
   than a guess.
3. Data-quality defects are injected at known rates, so the validation stage in
   Week 1 can be scored against the number of defects that were actually planted.

Ground-truth process (documented so feature engineering can be justified):

    road_km       = haversine_km * circuity
    speed_kmph    = free_flow * (1 - congestion_sensitivity * traffic_index)
                             * weather_speed_multiplier
                             * long_trip_bonus(road_km)
    duration_min  = road_km / speed_kmph * 60 + pickup_overhead
    duration_min *= lognormal_noise

``traffic_index`` itself is driven by an hour-of-day congestion profile, damped at
weekends. Everything downstream is a deterministic function of the seed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.config import ensure_parent, load_params, project_path
from src.utils.geo import haversine_km
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

GENERATOR_VERSION = "1.0.0"

# --------------------------------------------------------------------------------
# City geography: mixture-of-Gaussians pickup/drop-off hotspots across NYC.
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Hotspot:
    name: str
    lat: float
    lon: float
    weight: float
    spread_deg: float = 0.012


HOTSPOTS: tuple[Hotspot, ...] = (
    Hotspot("Midtown Manhattan", 40.7549, -73.9840, 0.22, 0.010),
    Hotspot("Financial District", 40.7075, -74.0113, 0.12, 0.008),
    Hotspot("Upper East Side", 40.7736, -73.9566, 0.10, 0.010),
    Hotspot("Upper West Side", 40.7870, -73.9754, 0.08, 0.010),
    Hotspot("Harlem", 40.8116, -73.9465, 0.05, 0.012),
    Hotspot("Williamsburg", 40.7081, -73.9571, 0.08, 0.011),
    Hotspot("Downtown Brooklyn", 40.6928, -73.9903, 0.07, 0.011),
    Hotspot("Long Island City", 40.7447, -73.9485, 0.07, 0.011),
    Hotspot("JFK Airport", 40.6413, -73.7781, 0.06, 0.004),
    Hotspot("LaGuardia Airport", 40.7769, -73.8740, 0.05, 0.004),
    Hotspot("South Bronx", 40.8296, -73.9262, 0.05, 0.013),
    Hotspot("South Brooklyn", 40.6501, -73.9496, 0.05, 0.013),
)

# Relative trip volume by hour of day (weekday vs weekend nightlife pattern).
DEMAND_WEEKDAY = np.array(
    [0.020, 0.012, 0.008, 0.006, 0.006, 0.010, 0.024, 0.048, 0.062, 0.052, 0.042, 0.043,
     0.046, 0.045, 0.047, 0.053, 0.064, 0.074, 0.070, 0.058, 0.048, 0.040, 0.034, 0.026]
)
DEMAND_WEEKEND = np.array(
    [0.052, 0.044, 0.034, 0.022, 0.012, 0.008, 0.010, 0.016, 0.026, 0.036, 0.044, 0.050,
     0.054, 0.054, 0.052, 0.050, 0.050, 0.052, 0.054, 0.056, 0.056, 0.058, 0.058, 0.052]
)

# Baseline road congestion by hour, 0 (free flow) to 1 (gridlock).
CONGESTION_BY_HOUR = np.array(
    [0.10, 0.07, 0.05, 0.05, 0.06, 0.10, 0.20, 0.42, 0.62, 0.66, 0.55, 0.50,
     0.52, 0.53, 0.56, 0.62, 0.72, 0.80, 0.78, 0.62, 0.45, 0.34, 0.26, 0.17]
)
WEEKEND_CONGESTION_DAMPING = 0.68
HOLIDAY_CONGESTION_DAMPING = 0.80

WEATHER_CONDITIONS = ("Clear", "Cloudy", "Rain", "Snow", "Fog")
WEATHER_SPEED_MULTIPLIER = {"Clear": 1.00, "Cloudy": 0.99, "Rain": 0.90, "Snow": 0.72, "Fog": 0.86}

# NYC monthly climatology: mean daily temperature (C) and precipitation/snow odds.
MONTHLY_TEMP_MEAN = np.array([1.0, 2.0, 6.0, 12.0, 18.0, 23.0, 26.0, 25.0, 21.0, 15.0, 9.0, 4.0])
MONTHLY_RAIN_PROB = np.array([0.16, 0.15, 0.20, 0.22, 0.24, 0.23, 0.24, 0.22, 0.20, 0.19, 0.18, 0.17])
MONTHLY_SNOW_PROB = np.array([0.14, 0.12, 0.05, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.09])
FOG_PROB = 0.03
CLOUD_PROB = 0.28

FREE_FLOW_KMPH = 34.0
CONGESTION_SENSITIVITY = 0.62
CIRCUITY_MEAN = 1.32
CIRCUITY_SD = 0.10
LOGNORMAL_NOISE_SIGMA = 0.16

PASSENGER_COUNTS = np.array([1, 2, 3, 4, 5, 6])
PASSENGER_WEIGHTS = np.array([0.62, 0.19, 0.07, 0.05, 0.04, 0.03])
VENDOR_IDS = np.array([1, 2, 3])
VENDOR_WEIGHTS = np.array([0.45, 0.40, 0.15])

RAW_COLUMNS = [
    "trip_id",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "store_and_fwd_flag",
    "weather_condition",
    "temperature_c",
    "precipitation_mm",
    "wind_kph",
    "traffic_index",
]


# --------------------------------------------------------------------------------
# Scenarios: the same generator drives both the clean training data and the
# Week 4 drift simulations, so a shift is a known intervention on the process.
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    traffic_shift: float = 0.0
    duration_multiplier: float = 1.0
    rain_prob_multiplier: float = 1.0
    precipitation_multiplier: float = 1.0
    extra_hotspots: tuple[Hotspot, ...] = ()
    speed_decay_per_day: float = 0.0
    demand_night_boost: float = 0.0


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario(
        name="baseline",
        description="Stationary process used for training and as the drift reference window.",
    ),
    "festival_surge": Scenario(
        name="festival_surge",
        description="City-wide event: congestion spikes and trips take materially longer.",
        traffic_shift=0.25,
        duration_multiplier=1.28,
        demand_night_boost=0.35,
    ),
    "monsoon": Scenario(
        name="monsoon",
        description="Sustained heavy-rain season shifting the weather covariate distribution.",
        rain_prob_multiplier=2.6,
        precipitation_multiplier=2.2,
        traffic_shift=0.08,
    ),
    "new_zone": Scenario(
        name="new_zone",
        description="Service expands into boroughs absent from training data (covariate shift).",
        extra_hotspots=(
            Hotspot("Staten Island", 40.5795, -74.1502, 0.12, 0.020),
            Hotspot("Far Rockaway", 40.6050, -73.7550, 0.08, 0.015),
        ),
    ),
    "concept_drift": Scenario(
        name="concept_drift",
        description=(
            "Road works degrade achievable speed a little each day: inputs look unchanged "
            "while the distance-to-duration relationship itself moves."
        ),
        speed_decay_per_day=0.0022,
    ),
}


def _sample_hotspots(
    rng: np.random.Generator, n: int, hotspots: tuple[Hotspot, ...]
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.array([h.weight for h in hotspots], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(hotspots), size=n, p=weights)
    centers_lat = np.array([h.lat for h in hotspots])[idx]
    centers_lon = np.array([h.lon for h in hotspots])[idx]
    spreads = np.array([h.spread_deg for h in hotspots])[idx]
    lat = centers_lat + rng.normal(0.0, spreads)
    lon = centers_lon + rng.normal(0.0, spreads) * 1.3  # longitude degrees are shorter at 40N
    return lat, lon


def _sample_pickup_datetimes(
    rng: np.random.Generator, n: int, start: str, end: str, scenario: Scenario
) -> pd.Series:
    calendar = pd.date_range(start=start, end=end, freq="D")
    dow = calendar.dayofweek.to_numpy()
    day_weight = np.array([0.130, 0.135, 0.140, 0.150, 0.170, 0.160, 0.115])[dow]
    day_weight = day_weight / day_weight.sum()
    day_idx = rng.choice(len(calendar), size=n, p=day_weight)
    chosen_days = calendar.to_numpy()[day_idx]

    is_weekend = np.isin(dow[day_idx], (5, 6))
    weekday_p = DEMAND_WEEKDAY / DEMAND_WEEKDAY.sum()
    weekend_p = DEMAND_WEEKEND / DEMAND_WEEKEND.sum()
    if scenario.demand_night_boost:
        night = np.zeros(24)
        night[18:24] = scenario.demand_night_boost
        night[0:3] = scenario.demand_night_boost
        weekday_p = weekday_p * (1.0 + night)
        weekend_p = weekend_p * (1.0 + night)
        weekday_p /= weekday_p.sum()
        weekend_p /= weekend_p.sum()

    hours = np.empty(n, dtype=np.int64)
    hours[~is_weekend] = rng.choice(24, size=int((~is_weekend).sum()), p=weekday_p)
    hours[is_weekend] = rng.choice(24, size=int(is_weekend.sum()), p=weekend_p)

    offsets = (
        hours * 3_600_000_000_000
        + rng.integers(0, 60, size=n) * 60_000_000_000
        + rng.integers(0, 60, size=n) * 1_000_000_000
    )
    return pd.Series(pd.to_datetime(chosen_days) + pd.to_timedelta(offsets, unit="ns"))


def _daily_weather(
    rng: np.random.Generator, calendar: pd.DatetimeIndex, scenario: Scenario
) -> pd.DataFrame:
    """Weather is drawn once per day, not per trip: trips on the same day share it."""
    month_idx = calendar.month.to_numpy() - 1
    p_snow = MONTHLY_SNOW_PROB[month_idx]
    p_rain = np.clip(MONTHLY_RAIN_PROB[month_idx] * scenario.rain_prob_multiplier, 0.0, 0.85)
    p_fog = np.full(len(calendar), FOG_PROB)
    p_cloud = np.full(len(calendar), CLOUD_PROB)
    p_clear = np.clip(1.0 - (p_snow + p_rain + p_fog + p_cloud), 0.01, None)

    probs = np.vstack([p_clear, p_cloud, p_rain, p_snow, p_fog]).T
    probs = probs / probs.sum(axis=1, keepdims=True)
    draws = rng.random(len(calendar))[:, None]
    condition_idx = (probs.cumsum(axis=1) < draws).sum(axis=1).clip(0, len(WEATHER_CONDITIONS) - 1)
    condition = np.array(WEATHER_CONDITIONS)[condition_idx]

    temperature = rng.normal(MONTHLY_TEMP_MEAN[month_idx], 4.5)
    temperature = np.where(condition == "Snow", np.minimum(temperature, 2.0), temperature)

    precipitation = np.zeros(len(calendar))
    rain_mask = condition == "Rain"
    snow_mask = condition == "Snow"
    precipitation[rain_mask] = rng.exponential(4.0, rain_mask.sum())
    precipitation[snow_mask] = rng.exponential(3.0, snow_mask.sum())
    precipitation *= scenario.precipitation_multiplier

    wind = rng.gamma(shape=4.0, scale=3.5, size=len(calendar))
    wind = np.where(condition.astype(str) == "Clear", wind * 0.8, wind)

    return pd.DataFrame(
        {
            "date": calendar.date,
            "weather_condition": condition,
            "temperature_c": np.round(temperature, 1),
            "precipitation_mm": np.round(np.clip(precipitation, 0.0, 78.0), 2),
            "wind_kph": np.round(np.clip(wind, 0.0, 115.0), 1),
        }
    )


def generate_trips(
    n_trips: int,
    start_date: str,
    end_date: str,
    seed: int,
    scenario: Scenario,
) -> pd.DataFrame:
    """Produce clean trips from the documented data-generating process."""
    rng = np.random.default_rng(seed)
    hotspots = HOTSPOTS + scenario.extra_hotspots

    pickup_dt = _sample_pickup_datetimes(rng, n_trips, start_date, end_date, scenario)
    pickup_lat, pickup_lon = _sample_hotspots(rng, n_trips, hotspots)
    dropoff_lat, dropoff_lon = _sample_hotspots(rng, n_trips, hotspots)

    calendar = pd.date_range(start=start_date, end=end_date, freq="D")
    weather = _daily_weather(rng, calendar, scenario)
    trips = pd.DataFrame({"date": pickup_dt.dt.date})
    trips = trips.merge(weather, on="date", how="left")

    hour = pickup_dt.dt.hour.to_numpy()
    dayofweek = pickup_dt.dt.dayofweek.to_numpy()
    is_weekend = np.isin(dayofweek, (5, 6))

    congestion = CONGESTION_BY_HOUR[hour]
    congestion = np.where(is_weekend, congestion * WEEKEND_CONGESTION_DAMPING, congestion)
    traffic_index = np.clip(
        congestion + scenario.traffic_shift + rng.normal(0.0, 0.075, n_trips), 0.02, 0.99
    )

    straight_km = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    circuity = np.clip(rng.normal(CIRCUITY_MEAN, CIRCUITY_SD, n_trips), 1.05, 1.90)
    road_km = np.maximum(straight_km * circuity, 0.15)

    weather_mult = trips["weather_condition"].map(WEATHER_SPEED_MULTIPLIER).to_numpy(dtype=float)
    long_trip_bonus = np.minimum(1.0 + 0.045 * road_km, 1.40)

    speed = FREE_FLOW_KMPH * (1.0 - CONGESTION_SENSITIVITY * traffic_index)
    speed *= weather_mult * long_trip_bonus

    if scenario.speed_decay_per_day:
        elapsed_days = (pickup_dt - pd.Timestamp(start_date)).dt.days.to_numpy()
        speed *= np.clip(1.0 - scenario.speed_decay_per_day * elapsed_days, 0.45, 1.0)

    speed = np.maximum(speed, 4.0)

    pickup_overhead = 1.2 + rng.exponential(0.9, n_trips)
    duration_min = road_km / speed * 60.0 + pickup_overhead
    duration_min *= np.exp(rng.normal(0.0, LOGNORMAL_NOISE_SIGMA, n_trips))
    duration_min *= scenario.duration_multiplier
    duration_min = np.maximum(duration_min, 1.0)

    frame = pd.DataFrame(
        {
            "trip_id": [f"T{i:09d}" for i in range(n_trips)],
            "vendor_id": rng.choice(VENDOR_IDS, size=n_trips, p=VENDOR_WEIGHTS),
            "pickup_datetime": pickup_dt,
            "dropoff_datetime": pickup_dt + pd.to_timedelta(duration_min, unit="m"),
            "passenger_count": rng.choice(
                PASSENGER_COUNTS, size=n_trips, p=PASSENGER_WEIGHTS
            ).astype(float),
            "pickup_latitude": np.round(pickup_lat, 6),
            "pickup_longitude": np.round(pickup_lon, 6),
            "dropoff_latitude": np.round(dropoff_lat, 6),
            "dropoff_longitude": np.round(dropoff_lon, 6),
            "store_and_fwd_flag": np.where(rng.random(n_trips) < 0.015, "Y", "N"),
            "weather_condition": trips["weather_condition"].to_numpy(),
            "temperature_c": trips["temperature_c"].to_numpy(),
            "precipitation_mm": trips["precipitation_mm"].to_numpy(),
            "wind_kph": trips["wind_kph"].to_numpy(),
            "traffic_index": np.round(traffic_index, 4),
        }
    )
    return frame.sort_values("pickup_datetime", ignore_index=True)[RAW_COLUMNS]


def inject_defects(
    frame: pd.DataFrame, defect_rates: dict[str, float], seed: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Corrupt a known fraction of rows so the validation stage has real work to do.

    Returns the corrupted frame and the exact planted-defect counts, which the
    validation report is scored against.
    """
    rng = np.random.default_rng(seed + 1)
    frame = frame.copy()
    n = len(frame)
    planted: dict[str, int] = {}

    def pick(rate: float) -> np.ndarray:
        size = int(round(n * rate))
        return rng.choice(n, size=size, replace=False) if size else np.array([], dtype=int)

    idx = pick(defect_rates.get("missing_gps", 0.0))
    if len(idx):
        half = len(idx) // 2
        frame.loc[frame.index[idx[:half]], ["pickup_latitude", "pickup_longitude"]] = np.nan
        frame.loc[frame.index[idx[half:]], ["dropoff_latitude", "dropoff_longitude"]] = np.nan
    planted["missing_gps"] = len(idx)

    idx = pick(defect_rates.get("invalid_timestamp", 0.0))
    if len(idx):
        # Dropoff recorded before pickup: a classic clock-skew / late-sync artefact.
        frame.loc[frame.index[idx], "dropoff_datetime"] = frame.loc[
            frame.index[idx], "pickup_datetime"
        ] - pd.to_timedelta(rng.integers(1, 90, len(idx)), unit="m")
    planted["invalid_timestamp"] = len(idx)

    idx = pick(defect_rates.get("out_of_bounds_gps", 0.0))
    if len(idx):
        frame.loc[frame.index[idx], "pickup_latitude"] = rng.choice([0.0, 90.0, -41.2], len(idx))
        frame.loc[frame.index[idx], "pickup_longitude"] = rng.choice([0.0, 180.0, -120.5], len(idx))
    planted["out_of_bounds_gps"] = len(idx)

    idx = pick(defect_rates.get("extreme_duration", 0.0))
    if len(idx):
        blow_up = rng.choice([0.05, 40.0, 90.0], len(idx))
        base = (
            frame.loc[frame.index[idx], "dropoff_datetime"]
            - frame.loc[frame.index[idx], "pickup_datetime"]
        ).dt.total_seconds() / 60.0
        frame.loc[frame.index[idx], "dropoff_datetime"] = frame.loc[
            frame.index[idx], "pickup_datetime"
        ] + pd.to_timedelta(base.to_numpy() * blow_up, unit="m")
    planted["extreme_duration"] = len(idx)

    idx = pick(defect_rates.get("impossible_speed", 0.0))
    if len(idx):
        # Same trip distance completed in a fraction of the time: teleporting vehicle.
        base = (
            frame.loc[frame.index[idx], "dropoff_datetime"]
            - frame.loc[frame.index[idx], "pickup_datetime"]
        ).dt.total_seconds() / 60.0
        frame.loc[frame.index[idx], "dropoff_datetime"] = frame.loc[
            frame.index[idx], "pickup_datetime"
        ] + pd.to_timedelta(np.abs(base.to_numpy()) * 0.02, unit="m")
    planted["impossible_speed"] = len(idx)

    idx = pick(defect_rates.get("missing_weather", 0.0))
    if len(idx):
        frame.loc[
            frame.index[idx],
            ["weather_condition", "temperature_c", "precipitation_mm", "wind_kph"],
        ] = np.nan
    planted["missing_weather"] = len(idx)

    idx = pick(defect_rates.get("missing_passenger_count", 0.0))
    if len(idx):
        frame.loc[frame.index[idx], "passenger_count"] = np.nan
    planted["missing_passenger_count"] = len(idx)

    n_dupes = int(round(n * defect_rates.get("duplicate_trip_id", 0.0)))
    if n_dupes:
        source = rng.choice(n, size=n_dupes, replace=False)
        duplicates = frame.iloc[source].copy()
        frame = pd.concat([frame, duplicates], ignore_index=True)
    planted["duplicate_trip_id"] = n_dupes

    return frame.sort_values("pickup_datetime", ignore_index=True), planted


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic NYC trip dataset.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--scenario", default=None, help="Overrides generate.scenario in params.")
    parser.add_argument("--n-trips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="Overrides the default raw output path.")
    parser.add_argument(
        "--clean", action="store_true", help="Skip defect injection (used for drift replays)."
    )
    args = parser.parse_args()

    params = load_params(args.params)
    gen = params["generate"]
    scenario_name = args.scenario or gen["scenario"]
    if scenario_name not in SCENARIOS:
        raise SystemExit(f"Unknown scenario {scenario_name!r}; choose from {sorted(SCENARIOS)}")
    scenario = SCENARIOS[scenario_name]

    n_trips = args.n_trips or gen["n_trips"]
    seed = args.seed if args.seed is not None else params["seed"]
    output = project_path(args.output or f"{params['paths']['raw']}/{gen['output_file']}")

    logger.info("Generating %s trips | scenario=%s | seed=%s", f"{n_trips:,}", scenario.name, seed)
    frame = generate_trips(n_trips, gen["start_date"], gen["end_date"], seed, scenario)

    if args.clean:
        planted: dict[str, int] = {}
    else:
        frame, planted = inject_defects(frame, gen["defect_rates"], seed)
        logger.info("Planted defects: %s", planted)

    ensure_parent(output)
    frame.to_parquet(output, index=False)

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scenario": scenario.name,
        "scenario_description": scenario.description,
        "seed": seed,
        "requested_trips": n_trips,
        "rows_written": int(len(frame)),
        "date_range": [gen["start_date"], gen["end_date"]],
        "columns": list(frame.columns),
        "planted_defects": planted,
        "defect_rates": {} if args.clean else gen["defect_rates"],
    }
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Wrote %s rows -> %s", f"{len(frame):,}", output)
    logger.info("Wrote generation metadata -> %s", meta_path)


if __name__ == "__main__":
    main()

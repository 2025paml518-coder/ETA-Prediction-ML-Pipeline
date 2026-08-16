"""The one and only feature pipeline.

Training-serving skew is the failure mode this module exists to prevent. Rather
than describing the transformations twice - once in a training notebook and once
in the API - both paths import ``FeaturePipeline`` and call the same
``transform``. The fitted state (zone clusters, speed priors, column order) is
persisted next to the model and reloaded by the service, so an inference request
travels through byte-identical code to a training row.

Fitted state is learned from the **training partition only**; the split stage runs
first precisely so that this is possible. That covers the three non-negotiable
requirements M2 2.6.4 places on imputation: the strategy is fixed before training, the
fill values come from training data alone, and they are persisted and reused at serving
rather than recomputed from production data.

Zone identity is deliberately expressed as centroid coordinates and learned speed
priors rather than raw cluster ids: the resulting matrix is fully numeric and
ordinal-meaningful, so a linear model and a gradient-boosted model can consume the
exact same feature matrix without per-model preprocessing branches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import load_params
from src.utils.calendar_nyc import is_holiday
from src.utils.geo import bearing_deg, haversine_km, manhattan_km

FEATURE_PIPELINE_VERSION = "1.0.0"

WEATHER_CATEGORIES = ("Clear", "Cloudy", "Rain", "Snow", "Fog")
WEATHER_SEVERITY = {"Clear": 0.0, "Cloudy": 1.0, "Fog": 2.0, "Rain": 3.0, "Snow": 4.0}
VENDOR_CATEGORIES = (1, 2, 3)

TARGET = "trip_duration_min"

IMPUTABLE_NUMERIC = ("temperature_c", "precipitation_mm", "wind_kph")

# Columns a caller must provide. Note the absence of dropoff_datetime: it is the
# target and is unavailable at inference time, so no feature may depend on it.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
    "vendor_id",
    "store_and_fwd_flag",
    "weather_condition",
    "temperature_c",
    "precipitation_mm",
    "wind_kph",
    "traffic_index",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    # geometry
    "haversine_km",
    "manhattan_km",
    "bearing_sin",
    "bearing_cos",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    # cyclical time
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    # calendar flags
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "is_holiday",
    # conditions
    "traffic_index",
    "temperature_c",
    "precipitation_mm",
    "wind_kph",
    "weather_severity",
    "weather_Clear",
    "weather_Cloudy",
    "weather_Rain",
    "weather_Snow",
    "weather_Fog",
    # trip metadata
    "passenger_count",
    "vendor_1",
    "vendor_2",
    "vendor_3",
    "store_and_fwd",
    "weather_imputed",
    "passenger_count_imputed",
    # learned zone signals
    "pickup_zone_lat",
    "pickup_zone_lon",
    "dropoff_zone_lat",
    "dropoff_zone_lon",
    "same_zone",
    "zone_hour_speed_prior",
    "route_speed_prior",
    "expected_duration_zone_hour",
    "expected_duration_route",
)


class FeaturePipelineNotFitted(RuntimeError):
    pass


@dataclass
class FeaturePipeline:
    """Stateful feature builder shared by the training pipeline and the API."""

    n_zone_clusters: int = 20
    rush_hours_morning: tuple[int, int] = (7, 10)
    rush_hours_evening: tuple[int, int] = (16, 19)
    random_state: int = 42

    kmeans: KMeans | None = field(default=None, repr=False)
    zone_centroids: np.ndarray | None = field(default=None, repr=False)
    zone_hour_speed: pd.DataFrame | None = field(default=None, repr=False)
    route_speed: pd.DataFrame | None = field(default=None, repr=False)
    zone_speed: pd.DataFrame | None = field(default=None, repr=False)
    imputation: dict = field(default_factory=dict, repr=False)
    global_speed: float = 20.0
    fitted_on_rows: int = 0
    _lookup_cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ---------------------------------------------------------------- fitting
    @classmethod
    def from_params(cls, params: dict | None = None) -> FeaturePipeline:
        params = params or load_params()
        cfg = params["features"]
        return cls(
            n_zone_clusters=cfg["n_zone_clusters"],
            rush_hours_morning=tuple(cfg["rush_hours_morning"]),
            rush_hours_evening=tuple(cfg["rush_hours_evening"]),
            random_state=params["seed"],
        )

    def fit(self, frame: pd.DataFrame) -> FeaturePipeline:
        """Learn imputation values, zone clusters and speed priors from training data."""
        self._require_columns(frame)
        if TARGET not in frame.columns:
            raise ValueError(f"fit() needs the target column {TARGET!r}")

        self._fit_imputation(frame)

        # Imputation is not applied here: none of the columns fitted below are
        # imputable, so the fill would be discarded work.
        coords = np.vstack(
            [
                frame[["pickup_latitude", "pickup_longitude"]].to_numpy(dtype=float),
                frame[["dropoff_latitude", "dropoff_longitude"]].to_numpy(dtype=float),
            ]
        )
        self.kmeans = KMeans(
            n_clusters=self.n_zone_clusters, random_state=self.random_state, n_init=5
        ).fit(coords)
        # Only the centroids are retained. Zone assignment is nearest-centroid, so the
        # fitted estimator adds nothing at inference time beyond a pickle dependency.
        self.zone_centroids = np.round(self.kmeans.cluster_centers_.astype(np.float64), 9)

        work = pd.DataFrame(
            {
                "pickup_zone": self._assign_zone(frame, "pickup"),
                "dropoff_zone": self._assign_zone(frame, "dropoff"),
                "hour": pd.to_datetime(frame["pickup_datetime"]).dt.hour.to_numpy(),
                "speed_kmph": self._observed_speed(frame),
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()

        self.global_speed = float(work["speed_kmph"].median())
        self.zone_hour_speed = (
            work.groupby(["pickup_zone", "hour"])["speed_kmph"].median().rename("speed").reset_index()
        )
        self.route_speed = (
            work.groupby(["pickup_zone", "dropoff_zone"])["speed_kmph"]
            .median()
            .rename("speed")
            .reset_index()
        )
        self.zone_speed = (
            work.groupby("pickup_zone")["speed_kmph"].median().rename("speed").reset_index()
        )
        self.fitted_on_rows = int(len(frame))
        return self

    # ------------------------------------------------------------- imputation
    def _fit_imputation(self, frame: pd.DataFrame) -> None:
        """Learn fill values from the training partition (M2 2.6.4)."""
        month = pd.to_datetime(frame["pickup_datetime"]).dt.month

        observed_weather = frame["weather_condition"].dropna()
        global_weather = str(observed_weather.mode().iat[0]) if not observed_weather.empty else "Clear"
        weather_by_month = (
            frame.dropna(subset=["weather_condition"])
            .groupby(month[frame["weather_condition"].notna()])["weather_condition"]
            .agg(lambda s: str(s.mode().iat[0]))
            .to_dict()
        )

        numeric_by_month: dict[str, dict[str, float]] = {}
        numeric_global: dict[str, float] = {}
        for column in IMPUTABLE_NUMERIC:
            numeric_global[column] = float(frame[column].median())
            numeric_by_month[column] = {
                str(int(k)): float(v)
                for k, v in frame.groupby(month)[column].median().dropna().to_dict().items()
            }

        self.imputation = {
            "weather_condition": {
                "strategy": "month_mode",
                "by_month": {str(int(k)): v for k, v in weather_by_month.items()},
                "global": global_weather,
            },
            "numeric": {
                "strategy": "month_median",
                "by_month": numeric_by_month,
                "global": numeric_global,
            },
            "passenger_count": {
                "strategy": "median",
                "global": float(frame["passenger_count"].median()),
            },
        }

    def _apply_imputation(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Fill repairable gaps using the persisted training-time values.

        Returns only the imputed columns rather than a copy of the whole frame: the
        other 8 input columns are read unchanged, and copying them per request was
        pure overhead.
        """
        if not self.imputation:
            raise FeaturePipelineNotFitted("Imputation values have not been fitted")

        month = pd.to_datetime(frame["pickup_datetime"]).dt.month.astype(str)

        weather_cfg = self.imputation["weather_condition"]
        weather = (
            frame["weather_condition"]
            .astype("object")
            .fillna(month.map(weather_cfg["by_month"]).fillna(weather_cfg["global"]))
        )

        # A request carrying an explicit null arrives as an object column, so these are
        # coerced before filling; without it the fill value lands in an object series
        # and every downstream numeric comparison silently changes behaviour.
        numeric_cfg = self.imputation["numeric"]
        imputed: dict[str, np.ndarray] = {"weather_condition": weather.to_numpy()}
        for column in IMPUTABLE_NUMERIC:
            fallback = numeric_cfg["global"][column]
            imputed[column] = (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(month.map(numeric_cfg["by_month"][column]).fillna(fallback))
                .to_numpy(dtype=float)
            )

        imputed["passenger_count"] = (
            pd.to_numeric(frame["passenger_count"], errors="coerce")
            .fillna(self.imputation["passenger_count"]["global"])
            .to_numpy(dtype=float)
        )
        return imputed

    # ------------------------------------------------------------- transform
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build the model-ready feature matrix in the canonical column order.

        Columns are accumulated in a plain dict of arrays and the DataFrame is built
        once. Assigning 49 columns onto a live frame instead cost a block-manager
        insert apiece, which was 44% of serving latency and did no useful work.
        """
        if self.zone_centroids is None:
            raise FeaturePipelineNotFitted("Call fit() or load() before transform()")
        self._require_columns(frame)

        # Missingness is recorded before it is filled: it is frequently informative,
        # and the indicator is what makes "indicator + fill" safe (M2 Table 2.3).
        weather_missing = frame["weather_condition"].isna().to_numpy().astype(float)
        passenger_missing = frame["passenger_count"].isna().to_numpy().astype(float)
        imputed = self._apply_imputation(frame)

        out: dict[str, np.ndarray] = {}
        pickup_dt = pd.to_datetime(frame["pickup_datetime"])

        p_lat = frame["pickup_latitude"].to_numpy(dtype=float)
        p_lon = frame["pickup_longitude"].to_numpy(dtype=float)
        d_lat = frame["dropoff_latitude"].to_numpy(dtype=float)
        d_lon = frame["dropoff_longitude"].to_numpy(dtype=float)

        distance = haversine_km(p_lat, p_lon, d_lat, d_lon)
        out["haversine_km"] = distance
        out["manhattan_km"] = manhattan_km(p_lat, p_lon, d_lat, d_lon)
        bearing = np.radians(bearing_deg(p_lat, p_lon, d_lat, d_lon))
        out["bearing_sin"] = np.sin(bearing)
        out["bearing_cos"] = np.cos(bearing)
        out["pickup_latitude"] = p_lat
        out["pickup_longitude"] = p_lon
        out["dropoff_latitude"] = d_lat
        out["dropoff_longitude"] = d_lon

        hour = pickup_dt.dt.hour.to_numpy()
        dow = pickup_dt.dt.dayofweek.to_numpy()
        month = pickup_dt.dt.month.to_numpy()
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

        morning = (hour >= self.rush_hours_morning[0]) & (hour < self.rush_hours_morning[1])
        evening = (hour >= self.rush_hours_evening[0]) & (hour < self.rush_hours_evening[1])
        weekend = np.isin(dow, (5, 6))
        out["is_weekend"] = weekend.astype(float)
        out["is_rush_hour"] = ((morning | evening) & ~weekend).astype(float)
        out["is_night"] = ((hour >= 22) | (hour < 5)).astype(float)
        out["is_holiday"] = is_holiday(pickup_dt).astype(float)

        weather_values = imputed["weather_condition"]
        out["traffic_index"] = frame["traffic_index"].to_numpy(dtype=float)
        out["temperature_c"] = imputed["temperature_c"]
        out["precipitation_mm"] = imputed["precipitation_mm"]
        out["wind_kph"] = imputed["wind_kph"]
        out["weather_severity"] = np.array(
            [WEATHER_SEVERITY.get(value, 0.0) for value in weather_values], dtype=float
        )
        for category in WEATHER_CATEGORIES:
            out[f"weather_{category}"] = (weather_values == category).astype(float)

        out["passenger_count"] = imputed["passenger_count"]
        vendor = frame["vendor_id"].to_numpy()
        for category in VENDOR_CATEGORIES:
            out[f"vendor_{category}"] = (vendor == category).astype(float)
        out["store_and_fwd"] = (
            frame["store_and_fwd_flag"].astype("object").fillna("N").to_numpy() == "Y"
        ).astype(float)
        out["weather_imputed"] = weather_missing
        out["passenger_count_imputed"] = passenger_missing

        pickup_zone = self._assign_zone(frame, "pickup")
        dropoff_zone = self._assign_zone(frame, "dropoff")
        centroids = self.zone_centroids
        out["pickup_zone_lat"] = centroids[pickup_zone, 0]
        out["pickup_zone_lon"] = centroids[pickup_zone, 1]
        out["dropoff_zone_lat"] = centroids[dropoff_zone, 0]
        out["dropoff_zone_lon"] = centroids[dropoff_zone, 1]
        out["same_zone"] = (pickup_zone == dropoff_zone).astype(float)

        keys = pd.DataFrame(
            {"pickup_zone": pickup_zone, "dropoff_zone": dropoff_zone, "hour": hour}
        )
        zone_hour_prior = self._lookup(keys, self.zone_hour_speed, ["pickup_zone", "hour"])
        route_prior = self._lookup(keys, self.route_speed, ["pickup_zone", "dropoff_zone"])
        out["zone_hour_speed_prior"] = zone_hour_prior
        out["route_speed_prior"] = route_prior

        with np.errstate(divide="ignore", invalid="ignore"):
            out["expected_duration_zone_hour"] = distance / zone_hour_prior * 60.0
            out["expected_duration_route"] = distance / route_prior * 60.0

        matrix = np.column_stack([out[column] for column in FEATURE_COLUMNS]).astype("float64")
        matrix[~np.isfinite(matrix)] = 0.0
        return pd.DataFrame(matrix, columns=list(FEATURE_COLUMNS), index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _require_columns(frame: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

    @staticmethod
    def _observed_speed(frame: pd.DataFrame) -> np.ndarray:
        if "avg_speed_kmph" in frame.columns:
            return frame["avg_speed_kmph"].to_numpy(dtype=float)
        distance = haversine_km(
            frame["pickup_latitude"],
            frame["pickup_longitude"],
            frame["dropoff_latitude"],
            frame["dropoff_longitude"],
        )
        return distance / frame[TARGET].to_numpy(dtype=float) * 60.0

    def _assign_zone(self, frame: pd.DataFrame, prefix: str) -> np.ndarray:
        """Nearest-centroid assignment, identical to ``KMeans.predict``."""
        coords = frame[[f"{prefix}_latitude", f"{prefix}_longitude"]].to_numpy(dtype=float)
        distances = ((coords[:, None, :] - self.zone_centroids[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def _lookup(self, keys: pd.DataFrame, table: pd.DataFrame, on: list[str]) -> np.ndarray:
        """Join a learned prior, falling back zone-level then global for unseen keys.

        Indexed reindex rather than a merge: a merge costs milliseconds of fixed
        overhead regardless of size, which dominated single-request serving latency
        while doing almost no work.
        """
        index = (
            pd.MultiIndex.from_arrays([keys[column].to_numpy() for column in on])
            if len(on) > 1
            else pd.Index(keys[on[0]].to_numpy())
        )
        lookup = self._indexed(table, on)
        speed = lookup.reindex(index).to_numpy(dtype=float)

        missing = np.isnan(speed)
        if missing.any():
            zone_lookup = self._indexed(self.zone_speed, ["pickup_zone"])
            zone_fallback = zone_lookup.reindex(
                pd.Index(keys["pickup_zone"].to_numpy())
            ).to_numpy(dtype=float)
            speed = np.where(missing, zone_fallback, speed)
            speed = np.where(np.isnan(speed), self.global_speed, speed)

        return np.clip(speed, 1.0, 120.0)

    def _indexed(self, table: pd.DataFrame, on: list[str]) -> pd.Series:
        """Cache the key-indexed view of a prior table; rebuilt only when it changes."""
        cache_key = id(table), tuple(on)
        cached = self._lookup_cache.get(cache_key)
        if cached is None:
            cached = table.set_index(on)["speed"]
            self._lookup_cache[cache_key] = cached
        return cached

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        priors = {
            "global_speed": self.global_speed,
            "zone_centroids": self.zone_centroids.tolist(),
            "imputation": self.imputation,
            "zone_hour_speed": self.zone_hour_speed.to_dict("records"),
            "route_speed": self.route_speed.to_dict("records"),
            "zone_speed": self.zone_speed.to_dict("records"),
        }
        (directory / "speed_priors.json").write_text(json.dumps(priors), encoding="utf-8")
        (directory / "feature_spec.json").write_text(
            json.dumps(self.feature_spec(), indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> FeaturePipeline:
        directory = Path(directory)
        spec = json.loads((directory / "feature_spec.json").read_text(encoding="utf-8"))
        if spec["feature_columns"] != list(FEATURE_COLUMNS):
            raise FeaturePipelineNotFitted(
                "Persisted feature_spec.json disagrees with the installed FEATURE_COLUMNS; "
                "the serving code and the trained artefact are out of sync."
            )
        priors = json.loads((directory / "speed_priors.json").read_text(encoding="utf-8"))
        pipeline = cls(
            n_zone_clusters=spec["n_zone_clusters"],
            rush_hours_morning=tuple(spec["rush_hours_morning"]),
            rush_hours_evening=tuple(spec["rush_hours_evening"]),
            random_state=spec["random_state"],
        )
        pipeline.zone_centroids = np.asarray(priors["zone_centroids"], dtype=np.float64)
        pipeline.imputation = priors["imputation"]
        pipeline.global_speed = priors["global_speed"]
        pipeline.zone_hour_speed = pd.DataFrame(priors["zone_hour_speed"])
        pipeline.route_speed = pd.DataFrame(priors["route_speed"])
        pipeline.zone_speed = pd.DataFrame(priors["zone_speed"])
        pipeline.fitted_on_rows = spec["fitted_on_rows"]
        return pipeline

    def feature_spec(self) -> dict:
        return {
            "version": FEATURE_PIPELINE_VERSION,
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "required_input_columns": list(REQUIRED_INPUT_COLUMNS),
            "target": TARGET,
            "n_zone_clusters": self.n_zone_clusters,
            "rush_hours_morning": list(self.rush_hours_morning),
            "rush_hours_evening": list(self.rush_hours_evening),
            "random_state": self.random_state,
            "fitted_on_rows": self.fitted_on_rows,
            "global_speed_kmph": round(self.global_speed, 4),
            "imputation_strategies": {
                key: value.get("strategy") for key, value in self.imputation.items()
            },
        }

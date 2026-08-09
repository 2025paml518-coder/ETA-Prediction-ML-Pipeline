# Design Decisions

The brief requires design decisions to be justified, not merely implemented. This
document records each decision, the alternatives considered, and why the alternative
was rejected. It grows week by week.

---

## Week 1 — M2: Data Engineering & Versioning

### D1. Synthetic dataset instead of Kaggle NYC Taxi Trip Duration

**Decision.** Generate trips from a documented data-generating process in
`src/data/generate_synthetic.py`.

**Alternatives considered.**

| Option | Why rejected |
| --- | --- |
| Kaggle NYC Taxi Trip Duration as-is | Contains no weather and no traffic columns, both of which the problem statement names explicitly. They would have to be fabricated and joined anyway. |
| Kaggle taxi data + real NOAA weather join | Adds credential management and a large binary download for no additional modelling insight, and still leaves traffic synthetic. |

**Why synthetic wins.** Week 4 requires *simulated drift*. If the generating process
is owned, a festival surge is a known intervention on a known parameter, and the drift
detector can be scored against ground truth. With a fixed public dump, "drift" can only
be approximated by resampling, which makes the monitoring evaluation circular.

A second benefit: defects are planted at known rates, so the validation stage can be
scored against the number of defects that actually exist (see
`reports/validation/validation_report.md`).

**Cost accepted.** Results are not comparable to published NYC taxi benchmarks. This is
acceptable because the rubric scores the *engineering system*, not leaderboard accuracy.

---

### D2. Fatal vs. repairable defects

**Decision.** Classify every data-quality failure into one of two buckets.

- **Fatal** — the target itself is untrustworthy: missing GPS, dropoff before pickup,
  out-of-bounds coordinates, impossible average speed, duplicate `trip_id`.
  These rows are written to `data/quarantine/` with a reason code.
- **Repairable** — the target is fine, only a covariate is missing: absent weather,
  absent passenger count. These are imputed from month-level statistics and the
  imputation is recorded in `weather_imputed` / `passenger_count_imputed`.

**Why.** Dropping every imperfect row would discard ~1.2% of otherwise valid training
signal for a missing wind-speed reading. Conversely, imputing a missing GPS coordinate
would invent the single most important feature in the model. The split is the point.

**Why quarantine rather than drop.** A dropped row is invisible. A quarantined row with
a reason code makes the failure *mix* auditable over time — and in Week 4 a shift in
that mix is itself a monitoring signal.

**Why imputation flags become features.** Missingness is rarely random. A trip whose
weather never arrived may correlate with an outage window that also affects duration,
so the flag is retained rather than discarded.

---

### D3. Pipeline aborts above a 15% quarantine rate

**Decision.** `validate` raises `ValidationFailure` when the quarantine rate exceeds
`validate.max_bad_row_fraction`, and when the batch is smaller than `min_rows`.

**Why.** Noise is expected and tolerable; a *step change* in the failure rate is not
noise, it means the upstream feed changed shape. Failing loudly at ingestion is far
cheaper than discovering it as unexplained model degradation four weeks later. This is
the "silent failure" mode named in CS 1 of the handout.

---

### D4. The validated table has an explicit schema contract

**Decision.** `src/data/schema.py` defines a pandera `DataFrameSchema` that the
validation stage asserts against its own output, and that the tests re-assert.

**Why.** The stage's guarantee to everything downstream is written down and executable.
If a future change to the repair logic lets a null through, validation fails on its own
output rather than propagating a null into feature engineering and, eventually, into a
served prediction.

---

### D5. Temporal split, executed before feature engineering

**Decision.** Split on `pickup_datetime` (70 / 15 / 15), and run the split *before*
the feature stage.

**Why temporal.** A random split places trips from the same rush hour, the same weather
system and the same road-works period on both sides of the boundary. The model would be
scored on conditions it had already seen, inflating offline metrics relative to
production, where prediction is always forward in time.

**Why before features.** The zone clusters and the speed priors are fitted statistics.
Fitting them on the full dataset would leak validation- and test-period information into
training features. Splitting first makes train-only fitting the natural implementation
rather than a discipline that has to be remembered.

---

### D6. Zone identity encoded as centroids and speed priors, not cluster ids

**Decision.** KMeans assigns each pickup and dropoff to one of 20 zones, but the feature
matrix carries zone *centroid coordinates*, a `same_zone` flag, and learned
zone-hour / route speed priors — never the raw integer cluster id.

**Why.** A cluster id is nominal. A linear model reading `pickup_zone = 17` would treat
it as seventeen times `pickup_zone = 1`, which is meaningless. One-hot encoding it would
force per-model preprocessing branches, and those branches are exactly where
training-serving skew breeds. Centroids and priors are numeric and ordinal-meaningful,
so a Ridge regression and a gradient-boosted tree consume the *identical* matrix, and a
tree can still recover zone boundaries by splitting on centroid coordinates.

**Leakage note.** The speed priors are a form of target encoding, computed on the
training partition only. With ~102k training rows across 20 zones x 24 hours, each cell
averages several hundred observations, so a row's own contribution to its prior is
negligible. Unseen combinations fall back zone-level, then global.

---

### D7. One `FeaturePipeline` object, shared by training and serving

**Decision.** A single fitted, serialisable class provides `fit` / `transform` /
`save` / `load`. The training stage and the API both import it and both load the same
artefact directory.

**Why.** Training-serving skew is the failure this design exists to prevent. The common
alternative — feature code in a training notebook, re-implemented in the service —
produces drift between two codebases that no test observes. Here there is only one
implementation, `feature_spec.json` pins the column order and is verified on load, and
`tests/test_skew.py` asserts that a single-row transform through a reloaded pipeline is
bit-identical to the batched training transform.

The `transform` path also never reads `dropoff_datetime`, which is enforced by a test:
that column is the target and does not exist at inference time.

---

### D8. DVC for data versioning, and as the orchestrator

**Decision.** `dvc.yaml` defines the four Week 1 stages with explicit `deps`, `params`
and `outs`. Data lives in the DVC remote; code and reports live in Git.

**Why DVC over ad-hoc scripts.** Stage-level dependency tracking means changing
`validate.max_bad_row_fraction` re-runs validation and everything after it, and nothing
before it. That is what makes "reproduce the tagged dataset version" a one-line
operation rather than a manual ritual.

**Why not Airflow or Prefect.** They solve scheduling and distributed execution, neither
of which this project has. Adding a scheduler would be infrastructure with no
corresponding requirement — and the retraining trigger in Week 4 can simply invoke
`dvc repro`.

**Why reports are `cache: false`.** Report directories are declared as uncached outputs
so DVC tracks them as stage products while leaving the files in Git, where a reviewer
can read them directly from the commit history.

---

### D9. Artefacts are deterministic JSON, not pickles

**Decision.** The feature pipeline persists `speed_priors.json` (including zone
centroids) and `feature_spec.json`. No estimator is pickled, and zone assignment is
implemented as nearest-centroid rather than `KMeans.predict`.

**How this was found.** Running `dvc repro --force` twice produced a different hash for
`models/feature_pipeline` even though the fitted centroids were numerically identical.
Refitting an estimator and re-pickling it does not yield stable bytes, so every rebuild
dirtied `dvc.lock` while nothing had actually changed — which destroys the signal that
the lock file is supposed to carry.

**Why the fix is an improvement rather than a workaround.**

1. `KMeans.predict` is exactly `argmin` of squared Euclidean distance to the centroids,
   so nothing is lost. `tests/test_features.py` asserts the two agree exactly.
2. The serving path no longer deserialises a pickle. Unpickling executes arbitrary code,
   so a model artefact from an untrusted or compromised store is a remote-code-execution
   vector; JSON removes that class of risk entirely.
3. The serving container no longer needs the scikit-learn version that trained the
   model, eliminating a whole category of version-skew failure on deploy.
4. The artefact is human-readable and diffable in review.

**Residual non-determinism, accepted.** `trips_raw.meta.json` and the validation report
embed a wall-clock generation timestamp. That is provenance rather than data, and it is
deliberately kept. The per-row `quarantined_at_utc` column was removed for the same
reason: a timestamp on every quarantined row made a *data* artefact unreproducible for
no analytical benefit, since the batch timestamp is already in the report.

---

### D10. Imputation moved out of validation and into the feature pipeline

**Decision.** Validation no longer fills missing weather or passenger counts. It leaves
those nulls intact and quarantines nothing for them. The feature pipeline learns the
fill values from the training partition, persists them in `speed_priors.json`, and
applies them inside `transform`.

**Why it changed.** The original design computed month-level medians from the whole
batch during validation, which runs *before* the split. That broke all three of the
requirements M2 2.6.4 calls non-negotiable:

1. the fill value was computed from validation and test rows, not training rows alone;
2. it was never persisted, so it could not be reused;
3. serving would therefore have had to recompute it from production data — the precise
   mechanism that introduces the distribution differences the chapter warns about.

**Consequence for the schema.** The validated table now declares the imputable
covariates as nullable. The contract is honest about what it guarantees: required
fields are present and in range, repairable fields may be absent and are the feature
pipeline's responsibility.

**Why the indicator is kept.** `weather_imputed` and `passenger_count_imputed` are
recorded before filling. Missingness is rarely random, so the flag is a feature in its
own right — the "indicator + fill" row of M2 Table 2.3, which is the only strategy the
table lists with no associated ML risk.

---

### D11. Level 3 statistical validation with a training baseline

**Decision.** A `profile` stage builds a baseline from the training partition:
per-feature mean, standard deviation, quantiles, null rate, category frequencies, and a
seeded 2,000-value reference sample per continuous column. `validate` accepts an
optional `--baseline` and compares each incoming batch against it.

**Why a stored sample rather than summary statistics alone.** Summary statistics
support a mean-shift check but not a Kolmogorov-Smirnov test, which needs the reference
distribution. Retaining a deterministic subsample keeps the artefact small while
allowing the actual two-sample test the chapter names.

**Why the baseline cannot be built from the batch under test.** A baseline computed
from the same rows it is judging can never show that those rows have moved. It is
therefore fitted on train only, which also means the first pass over a fresh dataset
has nothing to compare against — Level 3 reports "skipped" rather than inventing a
result.

**Thresholds, and why a p-value alone is not the trigger.** With ~100k baseline rows a
KS test reaches significance on differences far too small to act on: in the festival
surge trial, `wind_kph` produced a p-value of 3.7e-13 for a mean shift of 0.013
standard deviations. Significance is therefore necessary but not sufficient — a
minimum effect size must also be met before warning, and the decision to *fail* is
driven by shift magnitude: warn beyond one baseline standard deviation, fail beyond
two, exactly as M2 2.5.3 recommends. This directly addresses the maintainability
concern in the chapter's review questions: a validation layer that cries wolf on every
batch is worse than none.

**Validation of the design.** Against a `festival_surge` batch the ranking came out as
the intervention predicts — `avg_speed_kmph` (KS 0.52), `traffic_index` (0.40) and
`trip_duration_min` (0.28) all warned, while the four coordinate features passed. A
surge changes congestion, not geography.

---

### D12. Level 4 business rules

**Decision.** Four compound rules that no single-column check can express:

| Rule | What it catches |
| --- | --- |
| `BR_PRECIPITATION_WITHOUT_WET_WEATHER` | Rain measured while the sky is reported clear |
| `BR_SNOW_ABOVE_FREEZING` | Snow recorded well above freezing |
| `BR_PARTIAL_WEATHER_RECORD` | A weather join that half-succeeded |
| `BR_STATIONARY_LONG_TRIP` | The vehicle never moved, but the meter ran for an hour |

**Why they matter more than range checks.** Every value involved is individually legal,
so Levels 1 and 2 pass them without comment. These are the violations most likely to
indicate a real upstream defect rather than noise — a broken join, a mislabelled feed,
a unit change. The generator plants `inconsistent_weather` at a known rate specifically
so this level can be shown to work: 590 rows were caught in the tagged run.

---

### D13. Six data quality dimensions are measured, not assumed

**Decision.** `src/data/quality.py` measures each dimension of M2 Table 2.1 separately
and reports them in `reports/validation/data_quality_dimensions.md`.

**Why separately.** Collapsing everything into one "rows rejected" number destroys the
diagnostic value. A completeness problem and a uniqueness problem demand different
responses — one is an imputation strategy, the other an idempotency bug in the
upstream pipeline — so they are counted and reported apart.

**Accuracy is reported as unmeasurable, deliberately.** No rule can tell whether a
recorded duration is the duration that actually elapsed. Rather than omit the
dimension or fake a check, the report states that establishing accuracy requires a
sampling audit against raw GPS traces, and offers implied-speed plausibility as a
proxy. Naming the limit is more useful than hiding it.

**Timeliness is measured against the batch, not the clock.** Wall-clock age would make
the metric change on every run and break reproducibility. Record age is expressed
relative to the newest event in the batch instead.

---

### D14. Idempotent writes, and the batch ingestion pattern

**Decision.** Every stage writes to a staging path and then atomically replaces its
target (`src/utils/io.py`). Ingestion is batch.

**Why staging then swap.** A stage that writes directly to its target is not safe to
retry: a crash midway leaves a truncated file that the next stage cannot distinguish
from a complete one. `os.replace` is atomic within a filesystem, so a stage either
produces a whole output or leaves the previous one untouched — the pattern M2 2.4.1
prescribes.

**Why batch rather than streaming or CDC.** M2 2.4 recommends batch as the default and
evolving only when the business case is clear. ETA prediction is request-driven: the
model is retrained on accumulated history, and the features an inference request needs
(distance, time of day, weather, traffic) arrive with the request itself rather than
being pre-aggregated from an event stream. Streaming would add consumer groups, offset
tracking and state management for no freshness the system can use. CDC would be the
right answer if trip records were amended after the fact — a fare dispute revising a
duration — which is a plausible extension but not part of this dataset.

**Data model.** Parquet on a local filesystem with a DVC-managed remote, which is a
minimal Lakehouse-shaped arrangement: columnar files plus a versioned metadata layer
giving reproducible historical access. A relational store was not used because feature
construction is an OLAP workload — large columnar aggregations — and M2 2.3.1 is
explicit that running those against an OLTP schema is the wrong architecture.

**Feature store.** Not adopted. M2 2.10 puts the crossover at roughly 100+ models
across multiple teams; this project has one model and three engineers, where the shared
feature module plus serialised parameters gives the same consistency guarantee at a
fraction of the operational cost. The offline/online split is nonetheless mirrored in
miniature: `speed_priors.json` is a small offline artefact of pre-computed
zone-and-hour aggregates, loaded once by the service and used for every request.

---

## M2 requirement coverage

| M2 section | Requirement | Where |
| --- | --- | --- |
| 2.2 | Six data quality dimensions measured | `src/data/quality.py` |
| 2.3 | Data model choice justified | D14 |
| 2.4 | Ingestion pattern choice justified | D14 |
| 2.4.1 | Idempotent writes | `src/utils/io.py` |
| 2.5.1 | Level 1 schema validation | `src/data/schema.py` |
| 2.5.2 | Level 2 range and domain | `src/data/validate.py` |
| 2.5.3 | Level 3 statistical validation | `src/data/statistical_validation.py` |
| 2.5.4 | Level 4 business rules | `src/data/validate.py` |
| 2.5.4 | Fail loudly, stop the pipeline | `ValidationFailure` |
| 2.6.1 | Categorical encoding; target-encoding leakage | D6, `build_features.py` |
| 2.6.3 | Cyclical sin/cos temporal features | `build_features.py` |
| 2.6.4 | Imputation: train-only, persisted, reused | D10 |
| 2.7 | Training-serving skew prevention | D7, `tests/test_skew.py` |
| 2.8 | Feature store evaluated | D14 |
| 2.8.3 | Point-in-time correctness | D5 |
| 2.9 | Data lineage | DVC lock + generation metadata; MLflow in Week 2 |

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

## Week 2 — M3: Experimentation, Versioning & Reproducibility

### D15. The model is selected on validation, never on test

**Decision.** `select_best` ranks candidates by validation MAE and raises outright if
the configured selection partition is `test`.

**Why this is not pedantry.** The first draft of this stage picked the winner with
`min(results, key=lambda r: r["metrics_test"]["mae"])`. That single line converts the
reported test MAE from an unbiased estimate of production error into an optimistic
one, because the test partition has now influenced the model through the selection
step. The report would then be quoting a number that means something different from
what a reader assumes it means. Since the whole purpose of holding out a test set is
to answer "what will this cost in production", corrupting it costs more than any
modelling gain.

The guard is a raised exception rather than a comment, because the failure is silent
and the resulting numbers still look entirely plausible.

**Consequence.** LightGBM won on validation (3.474) and also happens to lead on test
(3.565); the discipline cost nothing here, which is the usual case. It matters on the
occasions when the ranking differs, and those are exactly the occasions you cannot
detect without it.

---

### D16. TimeSeriesSplit inside the hyperparameter search

**Decision.** `RandomizedSearchCV` uses `TimeSeriesSplit(n_splits=3)`.

**Why.** D5 split the data temporally so the model would never be scored on
conditions it had already seen. A default `cv=5` re-introduces exactly that leak one
level down: shuffled folds train on later trips to predict earlier ones, within the
training partition. The search would then select hyperparameters that exploit
information unavailable at inference, and the CV score would be optimistic in a way
the validation score would not reveal.

**Cost accepted.** TimeSeriesSplit trains on less data in early folds, so CV scores
are slightly pessimistic and the search is marginally noisier. A pessimistic estimate
of an honest quantity beats an optimistic estimate of a meaningless one.

---

### D17. Ridge is wrapped in a scaling pipeline

**Decision.** `Pipeline([("scaler", StandardScaler()), ("model", Ridge())])`, with the
search space addressing `model__alpha` and `model__solver`.

**Why.** The feature matrix mixes `pickup_latitude` (std 0.05), `hour_sin` (std 0.69)
and `expected_duration_route` (std 14.2). Ridge penalises the squared L2 norm of the
coefficients, which is scale-dependent, so on unscaled inputs an `alpha` of 1000
penalises the coefficient on latitude almost out of existence while barely touching
the duration prior. The model that results is not the model the grid intended to test.

**Why inside a Pipeline rather than scaling the table.** Two reasons. The scaler is
refitted on each CV fold's training portion, so fold scores are not contaminated by
statistics from the fold being scored. And the fitted scaler is serialised *with* the
estimator, so the serving path cannot use different scaling parameters from training —
the same argument as D7, applied to the model artefact instead of the feature
pipeline (M2 2.6.2, 2.7.2).

Tree models are scale-invariant and are tuned directly, without the wrapper.

---

### D18. A median baseline is a tracked candidate

**Decision.** `DummyRegressor(strategy="median")` is trained, logged and reported
alongside the real models.

**Why.** R² of 0.90 sounds conclusive until you ask what the alternative was. The
baseline answers it in the units the business cares about: predicting the median for
every trip gives 12.14 min test MAE, and LightGBM gives 3.56 min. The 70.6% reduction
is the actual return on operating a model, a serving container and a monitoring stack.
Without that row, the comparison table measures models against each other but never
against doing nothing.

It is also the cheapest possible regression test on the feature pipeline: if a future
change ever makes the learned models approach 12 minutes, something upstream has
broken, and `test_learned_models_beat_the_median_baseline` fails.

---

### D19. Reproducibility is verified, not asserted

**Decision.** `src/models/reproduce_run.py` reads a run from MLflow — parameters, tags
and dataset hashes only — refits, and compares against the originally logged metrics,
exiting non-zero beyond a tolerance.

**Why read only from the tracker.** "Our code is deterministic" is a weaker claim than
the rubric asks for. The question is whether the *logged record* is complete enough to
stand on its own months later. Reconstructing the run from MLflow alone tests exactly
that, and fails if a parameter was never logged.

**Why it fails loudly.** A near-match that is quietly accepted is how experiment logs
become untrustworthy. The script reports the maximum absolute delta and exits non-zero
past tolerance.

**Result.** The selected LightGBM run reproduces with a maximum delta of 0.0 across
all nine tracked metrics.

**Provenance recorded per run.** Git commit, branch, working-tree cleanliness, and the
DVC md5 of every dataset consumed. The dirty-tree flag matters: a run recorded from a
modified working tree is not identified by its commit hash alone, and the script warns
when reproducing one instead of pretending otherwise (M2 2.9).

---

### D20. MLflow tracks artefacts and a signature, not just scalars

**Decision.** Each run logs params, five metrics per partition, residual and
feature-importance plots, a model signature, an input example, and the provenance tags
above. The winner is registered in the MLflow Model Registry as `eta-predictor`.

**Why the signature and input example.** They pin the expected column names, order and
dtypes into the model artefact, so the Week 3 service can validate an incoming payload
against the model rather than trusting that its own construction of the feature frame
happens to match. This is the training-serving skew guard (D7) carried one layer
further, into the model itself.

**Why the registry.** It gives the serving layer a stable name and version to resolve
rather than a filesystem path to a `.pkl`, which is what makes the Week 4 retraining
trigger able to promote a new model without redeploying the API.

**Why `mlruns/` is not a DVC output.** MLflow owns and rewrites that store on every
run; declaring it as a stage output would put two tools in charge of the same
directory. DVC tracks the derived artefacts (`models/trained/`, reports, metrics)
instead.

---

### D21. Reported findings are generated from results

**Decision.** `compare.py` derives every statement in the report from the run results,
including unflattering ones — the train-to-validation spread is labelled as
overfitting when it exceeds 25% of train MAE.

**Why.** The previous version wrote fixed conclusions into the report before any model
had run: "Best model achieves < 12 minutes MAE on test set ✓", "Gradient boosting
outperforms baseline by 10%+". Those sentences would have appeared unchanged had the
model been catastrophically bad. A report that asserts its conclusions regardless of
the evidence is worse than no report, because it is presented as evidence.

---

## Week 3 — M4: Model Packaging, Deployment & Serving

### D22. Liveness and readiness are separate endpoints

**Decision.** `/health` answers whenever the process is up and deliberately does not
touch the model. `/ready` returns 503 until both the model and the feature pipeline
have loaded, and reports which of the two is missing.

**Why.** Conflating them breaks in both directions. If the single probe checks the
model, an orchestrator will kill and restart a perfectly healthy process because a
model artefact is missing — which restarting will not fix. If it only checks the
process, traffic gets routed to a container that cannot serve. The container
healthcheck polls `/ready`, because a running process with no model is not useful.

---

### D23. Validation at the edge mirrors the training-time contract

**Decision.** Pydantic bounds are read from the same `validate.bounds` block in
params.yaml that the ingestion stage uses, `extra="forbid"` rejects unknown fields,
and two Level 4 business rules from Week 1 are re-enforced on the request.

**Why the same bounds.** If serving accepts a latitude that training would have
quarantined, the model is being asked about a region it never saw. Accepting rows
training would have discarded is a form of training-serving skew, so the two
boundaries are driven from one configuration block rather than being written twice.

**Why `extra="forbid"`.** A caller who sends `trafic_index` instead of `traffic_index`
otherwise gets a silently defaulted feature and a confident, wrong ETA. The typo is
reported as a 422 instead.

**Why business rules again at the edge.** `BR_PRECIPITATION_WITHOUT_WET_WEATHER` and
the partial-weather rule catch a caller whose own weather lookup half-failed. Imputing
those gaps would hide a broken upstream integration behind a plausible number — the
silent failure M2 opens with.

**Why weather is optional but all-or-nothing.** The feature pipeline carries
imputation values learned from the training partition, so a caller with no weather
feed still gets a prediction, and the response says `weather_imputed: true` rather
than pretending the input was complete. A *partial* record is different: it signals a
malfunction rather than an absence.

---

### D24. The model is resolved from the registry, with a run fallback

**Decision.** The service loads the highest version of `eta-predictor` from the MLflow
Model Registry, falling back to the best run's artefact when no registry is reachable.

**Why the registry.** It gives the service a stable name to resolve instead of a
filesystem path to a pickle. That is what lets the Week 4 retraining trigger promote a
new version without rebuilding or redeploying the container.

**Why a fallback.** A container that cannot start without a reachable tracking server
is harder to demonstrate and harder to debug. The fallback keeps the image runnable
in isolation.

**Artefacts load once, at startup.** Loading per request would put tens of
milliseconds of deserialisation into every call. A warm-up prediction is issued during
startup so the first real request does not absorb lazy initialisation.

---

### D25. Every prediction is logged before Week 4 needs it

**Decision.** Served predictions go to SQLite with their features, model version and
latency; `/feedback` attaches the observed duration to the original row by request id.

**Why now rather than in Week 4.** Monitoring cannot be retrofitted onto traffic that
was never recorded. Building the log with the API means Week 4 starts with history
instead of an empty table.

**Why SQLite over JSONL.** The monitoring job queries by time window and joins
predictions to outcomes. Doing that over log lines is a worse reimplementation of a
database, and SQLite adds no service to the container.

**Why `/feedback` exists at all.** Without observed durations the service can only
watch its *inputs* drift. Actual outcomes are what turn drift detection into error
measurement, and they are what the Week 4 retraining trigger thresholds on.

**Logging never fails a request.** Write errors are logged and swallowed: a monitoring
outage must not become a serving outage.

---

### D26. Latency is reported as percentiles

**Decision.** `/metrics` exposes a Prometheus histogram, and `scripts/loadtest.py`
reports p50/p90/p95/p99 for single requests plus per-trip cost across batch sizes.

**Why not the mean.** The average latency of a service describes nobody's experience.
A caller with a timeout meets the tail, so p95 and p99 are the numbers that decide
whether the service is usable.

**Why measure batching separately.** `/predict/batch` featurises once and makes one
vectorised model call, so per-trip cost falls sharply with payload size. That gap is
the entire justification for the endpoint, and quoting a single latency figure would
conceal it.

**What measuring actually found.** The first benchmark returned 16.5 req/s at a p50 of
936ms, which is unusable. Profiling rather than guessing located the cost:

| Component | Before | After |
| --- | --- | --- |
| Feature transform | 20.5 ms | 5.9 ms |
| Prediction logging | 5.8 ms | 0.13 ms |
| Model inference | 3.7 ms | 1.8 ms |

Three findings, none of which were the obvious suspect:

1. `transform` assigned 49 columns onto a live DataFrame one at a time. Each
   assignment triggers a pandas block-manager insert, which reallocates; together they
   were 44% of serving latency while doing no arithmetic. Accumulating into a dict of
   arrays and constructing the frame once removed it. My first guess had been the
   four `merge` calls, which turned out to cost almost nothing — replacing them with
   indexed lookups moved the number by 1ms.
2. `_apply_imputation` deep-copied the entire request frame to fill five columns.
   It now returns only the imputed arrays.
3. The prediction log opened and closed a SQLite connection per request, and adding
   WAL pragmas made it *worse* because they then ran on every connect. A thread-local
   connection with the pragmas applied once took it from 5.8ms to 0.13ms.

End to end this took throughput from 16.5 to 45.5 req/s and p99 from 1508ms to 742ms.
The optimised transform was verified to reproduce the committed feature tables
bit-for-bit across all 145,000 rows, so no retraining was required — it is a pure
speed change, not a behaviour change.

**What remains.** At concurrency 16 the p50 of 332ms is dominated by queueing, not
work: the server handles ~22ms per request, and 16 in flight means most of the wait is
queue time. FastAPI runs synchronous endpoints in a threadpool, so CPU-bound
featurisation is serialised by the GIL regardless of thread count. Real concurrency
comes from more Uvicorn workers, each holding its own copy of the model — which is why
the report states these are single-worker figures rather than presenting them as a
ceiling.

---

### D27. Multi-stage container, non-root, model baked in

**Decision.** A builder stage installs into a virtualenv; the runtime copies only that
venv plus the application and artefacts, and runs as uid 10001.

**Why multi-stage.** `build-essential` is needed to install the dependencies and is a
liability in a shipped image. Copying just the virtualenv leaves compilers and headers
behind.

**Why non-root.** A compromise of the service should not also be root inside the
container. It costs one `useradd`.

**Why dependencies are copied before source.** Editing application code would
otherwise invalidate the slow dependency layer on every rebuild.

**Why the model is baked in rather than fetched at start.** The image becomes a single
versioned artefact that runs with no network, which is what makes the deployment
reproducible. The trade-off is that promoting a model requires a rebuild; the volume
mount in `compose.yaml` is the escape hatch when that matters.

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

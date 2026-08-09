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

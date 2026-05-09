---
marp: true
theme: rose-pine
paginate: true
class: 
  lead
  invert
headingDivider: 2
footer: Pycon 2025
math: mathjax
---
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

<!-- Intro Slide -->
<!-- _paginate: skip -->
# The 80/20 of ML Monitoring
### Pietro Mascolo

#### Bologna, 2026-05-28

&nbsp;
&nbsp;
&nbsp;
&nbsp;
![bg right:25%](./imgs/bg.jpg)


> What Actually Matters in Real ML Systems — And How to Implement It in Python
> http://bit.ly/4dfU5lf


<!-- Welcome and set the tone: this talk is not “intro to dashboards.” We’re going to cover the small number of things that actually matter in real ML monitoring. Emphasize that monitoring is not MLOps fluff — it's core to responsible modeling. -->

## What We’ll Cover

- What makes ML monitoring fundamentally different
- The 5 failure signals that matter
- A lean but robust monitoring architecture
- Python-first tools that implement it
- Deep dive: drift, skew, freshness, quality, degradation
- The strategic mistakes teams keep repeating

[Placeholder: architecture overview]

<!-- Frame the narrative: the goal is clarity and discipline. We will focus on what is universal across orgs, not on vendor-specific solutions. -->

## Why This Talk Exists

The traditional approach:

- log predictions
- build pretty dashboards (that no one will ever use fully)
- pray nothing breaks

But real systems fail due to distributional instability, not exceptions.

<!-- Goal: Show the minimal set of signals that predict >80% of real failures, and the architecture that supports them.

[Placeholder: iceberg diagram (“visible failures” vs “invisible drift”)] -->

<!-- Explain the empirical reality: the majority of ML production failures come from data issues and unnoticed drift. Most teams dramatically overestimate what dashboards catch and underestimate pipeline rot. Set expectation: this talk prioritizes conceptual rigor over tooling worship. -->

## Why Traditional Monitoring Fails for ML

ML changes because the world changes.

## Traditional monitoring - assumptions

stable requirements
deterministic logic
static invariants

## The real world of ML

stochastic spaces
shifting distributions
underspecified objectives

Different rules, different failure modes.

[Placeholder: dataset shifting timeline]

<!-- Give an example: fraud models degrade because fraudsters adapt, not because code breaks. Monitoring ML is not about uptime — it’s about epistemic uncertainty. -->
## The Real Sources of Failure

&nbsp;
&nbsp;
&nbsp;
&nbsp;

It’s rarely the model...
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;


## The Real Sources of Failure

* data pipeline schema drift
* training-serving mismatch
* feature staleness
* fundamental concept drift
* corrupted ingestion
* misaligned evaluation
* feedback loop instability



[Placeholder: anatomy of ML failure]

<!-- Lean into the “data-centric” message: Monitoring models is downstream — catching issues after impact. Monitoring data lets you catch issues before predictions degrade. -->
# The 5 Monitoring Signals That Actually Matter

## The 80/20 signals

&nbsp;

* Data Drift
* Training–Serving Skew
* Feature Staleness
* Performance Degradation
* Data Quality / Pipeline Breakage

&nbsp;
&nbsp;
&nbsp;
If you track these five well, you can catch the majority of high-impact failures.

## 1. Data Drift

Definition:
Distribution shift between training and production inputs.

Why it matters:
Even small shifts can have multiplicative downstream effects.

## Two categories:

Covariate drift (feature distributions shift)
Prior drift (class frequencies shift)

### Tools & signals:

KL divergence, PSI, KS test
Embedding drift (advanced)
Statistical consistency checks

[Placeholder: distribution comparison diagram]

<!-- Highlight the limitations of simple metrics like PSI, and suggest monitoring multiple drift signals. Advanced teams use representation models for multimodal or high-dimensional drift detection. -->
## 2. Training–Serving Skew

The silent killer.

Occurs when:

preprocessing logic differs
feature generation diverges
missing transformations in serving layer
categorical encodings mismatch
real-time features are delayed vs batch training features

Skew = models behaving unpredictably.

[Placeholder: “train vs serve pipeline” mismatch illustration]

<!-- Emphasize: this is the failure mode that embarrasses teams. Most skew happens because feature engineering is duplicated instead of centralized. -->
## 3. Feature Staleness

Your model expects dynamic signals.
Instead, it gets stale, cached, outdated values.

Detect via:

freshness timestamps
variance windows
categorical entropy checks
rolling statistical fingerprints

Staleness is subtle; drift metrics won’t catch it.

[Placeholder: flatline feature chart]

<!-- Make clear: freshness monitoring is the least glamorous but the most ROI. Most production failures in industry tie back to stale or lagged features. -->
## 4. Performance Degradation

This is the visible symptom of everything else going wrong.

Key nuances:

labels may be delayed
direct evaluation may be impossible
proxies may misrepresent ground truth
multi-objective systems degrade unevenly

Advanced techniques:

shadow models
counterfactual evaluation
model confidence drift
synthetic perturbation tests

[Placeholder: metric decay visualization]

<!-- Stress that performance monitoring is downstream. It’s necessary but not sufficient — without upstream monitoring, it’s too late. -->
## 5. Data Quality & Pipeline Integrity

This catches:

schema drift
missing columns
dtype mismatch
null spikes
impossible values
out-of-range values
duplicates
silent partial failures

Python tooling:

Great Expectations
Pandera
Pydantic

[Placeholder: data quality grid]

<!-- Explain that data quality issues often masquerade as “model issues” but are unrelated to ML. This is the foundation layer of monitoring. -->
## A Minimal But Robust Python Monitoring Stack

You only need:

Evidently → drift & data quality
River → online drift detectors
MLflow → performance + metric logging
FastAPI → monitoring endpoints / orchestration
Airflow / cron → run periodic checks
Object storage → baselines, stats, history

This stack is enough for most real applications.

[Placeholder: architecture diagram]

<!-- Stress that the power comes from simplicity and control. Large MLOps platforms often obscure what’s going on — bad for debugging. -->
Reference Architecture

[Placeholder: more detailed block diagram]

## Flow:

Batch ingestion
Data checks
Feature store
Online prediction
Drift + freshness checks
Metric logging
Alerts + dashboards
<!-- Highlight separation of concerns: Monitoring is not tied to model hosting. It should survive model replacement. -->
# Core Code Concepts
## Drift Check (Evidently)

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df)
results = report.as_dict()
```

## Feature Freshness

```python
lag = (now - feature_df.timestamp.max()).total_seconds()
if lag > MAX_LAG_SEC:
alert("Feature is stale: user_score")
```

## Skew Check

```python
assert train_pipeline.hash == serve_pipeline.hash
```

<!-- Stress that the point is the *workflow*, not the exact code. Show how checks are atomic and composable. Encourage attendees to treat each signal as a small orthogonal unit. -->
## Anti-Patterns to Avoid

❌ Monitoring hundreds of metrics → alert collapse
❌ Reacting to noise instead of signals
❌ Monitoring the model without monitoring the data
❌ “Just add a dashboard” culture
❌ No baselines → meaningless drift checks
❌ Hard-coded thresholds
❌ One-shot evaluations

Fix the root, not the symptoms.

[Placeholder: “alert fatigue” cartoon]

<!-- Deliver this slide with blunt clarity. Every advanced team has suffered these issues. You’re showing them how to skip the first two years of mistakes. -->
# Principles of Effective ML Monitoring

## Monitor data first, models second

Track signals, not dashboards
Baseline everything
Automate quickly, tune slowly
Prove actionability before visualization
Avoid black-box monitoring tools
Keep the system smaller than your ability to understand it
<!-- Close the conceptual loop. The monitoring system must be controllable, interpretable, and maintainable. -->
## What You Can Do This Week

Add drift detection to one model
Add freshness checks to your critical features
Align training & serving preprocessing
Introduce baselines for data quality tests
Stand up a simple monitoring API
Create a single “model health” page

Small wins compound.

<!-- Give them achievable actions. The goal is to convert the talk into momentum. -->
## Q&A

Thanks for listening.
Let’s dig deep.

[Placeholder final graphic]

<!-- Invite advanced questions. Encourage scenario-based discussion. -->
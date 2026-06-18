---
name: data-profiling
description: Dataset profiling and feature quality assessment
---

# Data Profiling Skill

## Objective

Analyze dataset structure, target behavior, and perform lightweight
feature quality assessment before feature selection.

The goal is to provide a fast, scalable, and production-grade profiling
pipeline for large banking datasets (millions of rows and tens of
thousands of columns) without loading the full dataset into memory.

---

## Inputs

* CSV or Parquet dataset
* Target column

---

## Implementation Requirements

### Data Processing

* Use Polars as the primary dataframe engine.
* Prefer `scan_parquet()` over `read_parquet()`.
* Use lazy evaluation whenever possible.
* Minimize memory usage.
* Avoid full dataset materialization.
* Use modular functions.
* Use structured logging.
* Use explicit file I/O.
* Compute profiling metrics on the FULL dataset.
* Do not sample rows unless explicitly requested by the user.

---

## CSV Compatibility Requirements

When profiling CSV datasets:

### Delimiter Detection

Do not assume comma-separated files.

Automatically detect the delimiter from the input file before reading.

Use the detected delimiter consistently for all CSV reads.

---

### Schema Inference

Do not rely on default Polars schema inference.

Always use:

```python
pl.scan_csv(
    path,
    separator=delimiter,
    infer_schema_length=None
)
```

This ensures datatype inference uses the full dataset and avoids
incorrect type assignments.

---

### CSV Parse Recovery

If Polars raises datatype parsing errors during schema resolution
or collect():

```text
could not parse value as dtype ...
```

the script must automatically retry using:

```python
pl.scan_csv(
    path,
    separator=delimiter,
    infer_schema_length=None,
    ignore_errors=True
)
```

Profiling jobs should prioritize successful dataset ingestion over
strict typing.

---

### Large CSV Files

For very large CSV datasets:

* Prefer lazy scanning (`scan_csv`)
* Avoid full dataframe materialization
* Avoid Pandas unless explicitly required
* Use Polars lazy execution for all profiling metrics

---

## Script Requirements

The profiling script must:

* Define `WORKING_DIR` at the top.
* Use absolute paths.
* Include try/except blocks.
* Include structured logging.
* Print confirmation of saved files.
* Be saved under:

```text
WORKING_DIR/scripts/
```

* Write outputs only under:

```text
WORKING_DIR/data_profiling/
```

* Never write outputs under `BASE_WORKING_DIR`.

### Helper Scripts

Helper scripts are allowed when required for scalability or
execution constraints.

Store helper scripts under:

```text
WORKING_DIR/scripts/helpers/
```

Examples:

```text
scripts/helpers/compute_missing_counts.py
scripts/helpers/compute_target_stats.py
scripts/helpers/profile_metadata.py
```

The final orchestration script must remain:

```text
scripts/data_profiling.py
```

---

## Polars Compatibility Requirements

Generated code must work across commonly deployed Polars versions.

Preferred schema access:

```python
lf = pl.scan_parquet(path)

schema = lf.schema

columns = list(schema.keys())

dtypes = dict(schema.items())
```

Preferred row count:

```python
n_rows = (
    lf.select(pl.len())
      .collect()
      .item()
)
```

Use:

```python
pl.len()
```

instead of:

```python
pl.count()
```

Avoid:

```python
pl.Schema
collect_schema()
schema.names()
pl.datatypes.is_numeric()
```

Avoid version-specific APIs.

Prefer stable Polars expressions.

### Additional Compatibility Rules

Avoid:

```python
pl.scan_parquet(path, columns=["col1", "col2"])
```

because older Polars versions may not support the `columns` argument.

Use:

```python
pl.scan_parquet(path).select(["col1", "col2"])
```

instead.

Avoid:

```python
pl.cut(...)
```

unless compatibility has been verified.

Avoid:

```python
quantile(..., interpolation=...)
```

unless compatibility has been verified.

Prefer version-independent implementations.

Schema resolution is allowed during profiling.

Warnings such as:

```text
PerformanceWarning: Resolving schema of LazyFrame
```

are acceptable and must not trigger repair workflows,
script regeneration, or reruns.

---

## Large Dataset Handling

Must support:

```text
Rows     : 1M+
Columns  : 10K–20K+
Size     : 10GB+
```

without exhausting memory.

---

## Full Dataset Metrics

The following MUST be computed on the FULL dataset without loading
it fully into memory:

* Row count
* Column count
* Datatype summary
* Target distribution
* Bad rate
* Class imbalance ratio
* Missing count per feature
* Missing percentage per feature

---

## Column Pruning

Only scan required columns.

Example:

```python
lf.select(["target"])
```

instead of scanning all columns.

---

## Missing Value Analysis

### Preferred Order

#### Option 1 (Preferred)

Use Parquet metadata null-count statistics.

Example:

```python
row_group.column(i).statistics.null_count
```

Benefits:

* Does not read data pages
* Extremely fast
* Suitable for datasets with 10K–20K+ columns
* Uses full dataset metadata

Whenever Parquet metadata provides reliable null counts, use
metadata-derived missing counts instead of scanning the dataset.

---

#### Option 2 (Fallback)

Use lazy Polars aggregation only when metadata null counts are
unavailable.

Example:

```python
BATCH_SIZE = 500

for batch in chunks(columns, BATCH_SIZE):

    result = (
        lf.select([
            pl.col(c).is_null().sum().alias(c)
            for c in batch
        ])
        .collect()
    )
```

Requirements:

* Batch columns
* Avoid extremely large query plans
* Avoid scanning all columns simultaneously

---

### Important Rule

Avoid scanning the entire dataset solely for missing counts when
metadata already contains reliable null_count statistics.

Metadata-first profiling is preferred.

---

## Memory Management

Avoid:

* Full dataframe materialization
* Loading dataset into Pandas
* Row-wise operations
* Python loops over records
* Repeated full dataset scans

Prefer:

* Lazy execution
* Aggregations
* Column pruning
* Metadata-based profiling
* Batched processing

---

## Progress Reporting

Long-running profiling jobs must emit structured progress events.

Use:

```python
def log_progress(stage, current, total, **kwargs):
    payload = {
        "event": "progress",
        "stage": stage,
        "current": current,
        "total": total,
        "percent_complete": round(
            current * 100 / total, 2
        ) if total else 100.0,
    }

    payload.update(kwargs)

    print(
        json.dumps(payload),
        flush=True,
    )
```

Progress events must be emitted as standalone JSON lines.

Example:

```json
{
  "event": "progress",
  "stage": "missingness_scan",
  "current": 500,
  "total": 15000,
  "percent_complete": 3.33
}
```

### Required Progress Points

Emit progress during:

* Dataset discovery
* Schema loading
* Row count computation
* Target analysis
* Metadata missingness analysis
* Batched missingness scan
* Feature quality generation
* Output generation

### Long Loop Rules

Progress reporting must not be limited to stage start/end.

For operations processing:

* parquet files
* row groups
* batches
* columns
* features

emit progress periodically.

Recommended frequency:

* every file
* every row group
* every batch
* every 500 columns
* every 100 features

or at a frequency that produces a visible update every 30–60 seconds.

Long-running loops must emit intermediate progress updates.

Avoid patterns such as:

```text
current=0,total=1
current=1,total=1
```

unless the operation completes within a few seconds.

Progress events should reflect actual work completed rather than
only stage boundaries.

The goal is to provide continuous execution visibility while a job
is RUNNING.

Progress events are informational only.

They must never be used to determine:

* Success
* Failure
* Completion

Success and failure are determined exclusively through:

* wait_for_job()
* validate_output_files()

Scripts should continue to emit normal INFO/WARNING/ERROR logs in
addition to progress events.

---

## Tasks

### Dataset Overview

Compute:

* Row count
* Column count
* Datatype summary

---

### Missing Value Analysis

For every feature compute:

* Missing count
* Missing percentage

Requirements:

* Full dataset
* Metadata-first approach
* Batched lazy scan fallback

---

### Target Analysis

Compute on FULL dataset:

* Target distribution
* Bad rate
* Class imbalance ratio

Only scan target column.

Example:

```python
target_stats = (
    lf.select(pl.col(target_col))
      .group_by(target_col)
      .agg(pl.len())
      .collect()
)
```

---

### Feature Quality Assessment

For every feature compute:

* feature
* dtype
* missing_count
* missing_pct

No additional statistics required.

---

## Initial Feature Reduction

Apply ONLY:

```text
missing_pct > 95%
```

Drop those features.

Retain all others.

Do NOT automatically remove:

* IDs
* Customer keys
* Account numbers
* Product identifiers
* Dates
* High-cardinality features
* Business variables

---

## Excluded Calculations

Do NOT compute:

* Duplicate analysis
* Unique count
* Cardinality
* Variance
* Near-zero variance
* Numeric summary
* Outlier analysis
* Correlation analysis

These belong to later stages.

---

## Outputs

### profiling_summary.json

Contains:

* dataset_type
* file_count
* dataset_size_gb
* target_column
* problem_type
* row_count
* column_count
* input_feature_count
* retained_feature_count
* dropped_feature_count
* target_distribution
* bad_rate
* class_imbalance_ratio

---

### feature_quality_report.csv

Columns:

```text
feature
dtype
missing_count
missing_pct
retained_flag
```

---

### preselected_features.csv

Retained features after missing-value filtering.

---

### profiling_report.md

Contains:

* Dataset Overview
* Target Summary
* Missing Value Summary
* Feature Reduction Summary

---

## Validation

Ensure:

* Dataset not empty
* Target exists
* Full dataset used for target analysis
* Full dataset used for missing-value analysis
* Output files generated
* Output files non-empty
* preselected_features.csv not empty

Raise explicit errors if validation fails.

---

## Failure Diagnostics

When execution fails:

* Raise explicit exceptions
* Include failing operation details
* Include relevant column names
* Include relevant file names

Preferred:

```python
raise ValueError(
    f"Target column '{target_col}' not found"
)
```

instead of:

```python
raise Exception("Failed")
```

Use:

```python
logging.exception(...)
```

for failure handling so stack traces appear in execution logs.

All exceptions should provide enough information for automatic repair
without requiring source-code inspection.

---

## Execution Reliability

Generated code must:

* Run without manual intervention
* Use Polars lazy execution
* Prefer metadata-based profiling
* Avoid loading full dataset into memory
* Use batched fallback scans
* Fail fast with clear errors
* Validate outputs before completion
* Produce deterministic results

---

## Expected Flow

```
Dataset
    ↓
Schema Scan
    ↓
Target Analysis
    ↓
Metadata Missingness Analysis
    ↓
Fallback Batched Missingness Scan (if needed)
    ↓
Drop Missing > 95%
    ↓
preselected_features.csv
    ↓
Reports
```

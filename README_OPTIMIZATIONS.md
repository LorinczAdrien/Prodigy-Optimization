# Prodigy Optimization Notes

This document describes all changes made to the original [peaclab/Prodigy](https://github.com/peaclab/Prodigy) implementation and the rationale behind each decision.

## Summary of Changes

| File | Status | Key Changes |
|------|--------|-------------|
| `src/config.py` | **NEW** | Centralised configuration constants |
| `src/vae.py` | **Optimized** | Random seeds, vectorized prediction, removed `disable_eager_execution` |
| `src/data_pipeline.py` | **Optimized** | Parallel TSFresh (`n_jobs=-1`), config constants |
| `src/reproducibility_experiments.py` | **Optimized** | Random seeds, `random_state=42`, config constants, memory cleanup |
| `src/reproducibility_plots.py` | **Fixed** | Undefined `verbose` variable bug |
| `src/anomaly_detector.py` | **Optimized** | Config constants, vectorized prediction |
| `src/constants.py` | Unchanged | Identical to original |
| `src/utils.py` | Unchanged | Identical to original |
| `src/ai4hpc_predict.py` | Unchanged | Identical to original |

---

## Detailed Changes

### 1. New File: `src/config.py`

**Purpose**: Centralise all magic numbers and tuneable parameters.

All hardcoded constants have been extracted to this file:
- `RANDOM_SEED = 42` — single source of truth for all random seeds
- `DEFAULT_EPOCHS = 1000`, `DEFAULT_BATCH_SIZE = 32` — training hyperparameters
- `THRESHOLD_PERCENTILE = 99` — VAE anomaly threshold percentile
- `INTERMEDIATE_DIM_RATIO = 2`, `LATENT_DIM_RATIO = 3` — VAE architecture ratios
- `VALIDATION_SPLIT = 0.1`, `LEARNING_RATE = 1e-4` — additional hyperparameters
- `N_JOBS = -1` — enables parallel processing in TSFresh (use all CPU cores)
- `HEALTHY_TEST_PERCENTAGES` — maps experiment config number to test fraction

---

### 2. `src/vae.py`

#### a) Random seed initialisation (reproducibility)

```python
# Added at module level, before any model construction
import random
import numpy as np
import tensorflow as tf

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)
```

**Why**: Without fixed seeds the VAE's weight initialisation, dropout, and
epsilon sampling in the reparameterisation trick all produce different values
on every run, making results non-reproducible.

#### b) Removed `disable_eager_execution()`

The original code called `disable_eager_execution()` unconditionally. In
TensorFlow 2.x this disables a large number of performance improvements and
is not required for the VAE to train correctly. It is commented out with an
explanation of when to re-enable it.

#### c) Vectorized prediction (performance)

```python
# Before (Python loop):
pred = [1 if curr_mae > self.threshold else 0 for curr_mae in mae_data]

# After (vectorized NumPy):
pred = (mae_data > self.threshold).astype(int)
```

**Why**: NumPy vectorised comparison is significantly faster than a Python
`for` loop for large arrays.  The output is identical (array of 0/1 integers).

#### d) Config constant for threshold percentile

```python
# Before:
self.threshold = np.percentile(mae_train.values, 99)

# After:
self.threshold = np.percentile(mae_train.values, config.THRESHOLD_PERCENTILE)
```

---

### 3. `src/data_pipeline.py`

#### a) Parallel TSFresh processing (performance)

```python
# Before (roll_time_series):
# n_jobs=-1,   # commented out

# After:
n_jobs=config.N_JOBS,   # -1 = use all CPU cores
```

```python
# Before (extract_features):
data_fe = extract_features(data, ...)  # no n_jobs

# After:
data_fe = extract_features(data, ..., n_jobs=config.N_JOBS)
```

**Why**: TSFresh feature extraction is the main computational bottleneck.
Enabling parallel processing can reduce wall-clock time proportionally to the
number of available CPU cores.

**Compatibility**: The extracted features are identical regardless of
parallelism level because TSFresh uses deterministic algorithms per series.

#### b) Import `config`

Added `import config` to use the centralised constants.

---

### 4. `src/reproducibility_experiments.py`

#### a) Random seed initialisation at the top of `main()`

```python
import random, numpy as np, tensorflow as tf, config

def main(...):
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
```

**Why**: Ensures every call to `main()` starts from the same random state,
making the entire experiment pipeline reproducible.

#### b) `random_state` in `train_test_split`

```python
# Before:
train_label_healthy, test_label_healthy = train_test_split(
    healthy_labels, test_size=healthy_test_data_percentage)

# After:
train_label_healthy, test_label_healthy = train_test_split(
    healthy_labels, test_size=healthy_test_data_percentage,
    random_state=config.RANDOM_SEED)
```

**Why**: Without `random_state` the split changes on every run, making
results non-reproducible regardless of NumPy seeds.

#### c) Use `config.HEALTHY_TEST_PERCENTAGES`

Replaced the long `if/elif` chain with a single dictionary lookup:

```python
# Before: 6-branch if/elif
if expConfig_num == 0:
    healthy_test_data_percentage = 0.2
elif expConfig_num == 1:
    ...

# After:
healthy_test_data_percentage = config.HEALTHY_TEST_PERCENTAGES[expConfig_num]
```

#### d) Use config constants for model hyperparameters

```python
intermediate_dim = int(input_dim / config.INTERMEDIATE_DIM_RATIO)
latent_dim      = int(input_dim / config.LATENT_DIM_RATIO)

vae.fit(...,
    epochs=config.DEFAULT_EPOCHS,
    batch_size=config.DEFAULT_BATCH_SIZE,
    validation_split=config.VALIDATION_SPLIT,
    ...)

vae = VAE(..., learning_rate=config.LEARNING_RATE)
```

#### e) Memory cleanup after each iteration

```python
del x_train_fe, x_test_fe, x_train_scaled, x_test_scaled
```

**Why**: These large DataFrames are not needed after the current iteration.
Explicit deletion allows Python's garbage collector to reclaim memory before
the next iteration begins, reducing peak memory usage in long runs.

---

### 5. `src/reproducibility_plots.py`

#### Bug fix: undefined `verbose` variable

**Original code** (lines 57, 66, 96):

```python
def main(results_dir, plot_output_dir):   # verbose NOT a parameter
    ...
    except:
        if verbose:   # NameError: name 'verbose' is not defined
            print(...)
```

The `verbose` variable was referenced inside `main()` but was never passed as
a parameter, causing a `NameError` at runtime whenever a result file was
missing.

**Fix**:

```python
def main(results_dir, plot_output_dir, verbose=False):  # added parameter
```

And at the call site:

```python
main(results_dir, plot_output_dir, verbose)  # pass the variable
```

#### Better exception handling

Replaced bare `except:` with `except (FileNotFoundError, IOError):` to avoid
accidentally silencing unexpected errors.

---

### 6. `src/anomaly_detector.py`

#### a) Use config constants for architecture ratios

```python
intermediate_dim = int(input_dim / config.INTERMEDIATE_DIM_RATIO)
latent_dim      = int(input_dim / config.LATENT_DIM_RATIO)
```

#### b) Vectorized prediction in `_predict_anomaly`

```python
pred = (mae_data > self.threshold).astype(int)
```

---

## Reproducibility Guarantees

### What is guaranteed to be identical

- Feature extraction results (TSFresh with fixed parameters)
- Train/test split (fixed `random_state=42`)
- VAE weight initialisation (fixed TF/NumPy seeds)
- Anomaly threshold (99th percentile, unchanged)
- Plot parameters and layout (unchanged)

### What may still differ

- **GPU non-determinism**: Certain GPU operations (cuDNN convolutions,
  atomics) may produce slightly different floating-point results on different
  hardware or driver versions, even with fixed seeds. This is a known
  limitation of GPU training with TensorFlow.
- **PYTHONHASHSEED**: Python's string/object hashing is randomised by
  default. Set `PYTHONHASHSEED=42` in your shell before running to fix it:
  ```bash
  export PYTHONHASHSEED=42
  python reproducibility_experiments.py
  ```

### How to verify output matches original

1. Run the original implementation with fixed seeds added manually.
2. Run this optimized implementation.
3. Compare JSON result files:
   ```bash
   diff original_results/expConfig_0_repeatNum_0_testResults.json \
        optimized_results/expConfig_0_repeatNum_0_testResults.json
   ```
4. Compare plots visually or with a pixel-diff tool.

---

## Performance Benchmarks

Benchmarks depend heavily on hardware (CPU cores, RAM, GPU). The primary
expected improvement is from enabling TSFresh parallel processing:

| Configuration | Original | Optimized |
|--------------|---------|----------|
| Feature extraction (n_jobs) | 1 (serial) | -1 (all cores) |
| Expected speedup | baseline | ~Nx (N = CPU cores) |

Prediction loop vectorization provides a minor improvement for large test
sets.

---

## Known Limitations

1. `disable_eager_execution()` is commented out. If you observe training
   instability in TF 2.6, uncomment it in `vae.py`.
2. GPU determinism cannot be fully guaranteed. Use CPU-only training for
   exact reproducibility across machines.
3. The `PYTHONHASHSEED` environment variable must be set externally before
   the Python interpreter starts — it cannot be set inside the script.

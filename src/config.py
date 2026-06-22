"""Configuration constants for reproducibility and optimization.

This module centralises all magic numbers and tuneable parameters so they
can be changed in one place rather than scattered across multiple files.
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 32
THRESHOLD_PERCENTILE = 99
INTERMEDIATE_DIM_RATIO = 2   # input_dim / INTERMEDIATE_DIM_RATIO
LATENT_DIM_RATIO = 3         # input_dim / LATENT_DIM_RATIO
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 1e-4

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
# N_JOBS=-1 uses all available CPU cores for parallel TSFresh processing.
# Set to 1 to disable parallelism (useful for debugging or determinism).
N_JOBS = -1


def get_tsfresh_n_jobs():
    """Return a TSFresh-compatible n_jobs value derived from N_JOBS.

    TSFresh does not accept sklearn-style ``-1`` as "all cores", so map it to
    the detected CPU count.
    """
    if N_JOBS == -1:
        return max(1, os.cpu_count() or 1)

    if isinstance(N_JOBS, int) and N_JOBS >= 0:
        return N_JOBS

    raise ValueError(
        f"Invalid N_JOBS={N_JOBS}. Use -1 for all cores or a non-negative integer."
    )

# ---------------------------------------------------------------------------
# Data split ratios for reproducibility experiments
# ---------------------------------------------------------------------------
# Maps expConfig_num -> fraction of healthy samples used as test data.
HEALTHY_TEST_PERCENTAGES = {
    0: 0.2,
    1: 0.4,
    2: 0.6,
    3: 0.8,
    4: 0.9,
    5: 0.95,
}

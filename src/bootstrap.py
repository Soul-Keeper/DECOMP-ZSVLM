import numpy as np


def _segment_sums(values, segments):
    keys, index = np.unique(segments, return_inverse=True)
    sums = np.bincount(index, weights=values, minlength=len(keys))
    counts = np.bincount(index, minlength=len(keys))
    return sums, counts


def cluster_ci(values, segments, n_boot=10000, seed=0, alpha=0.05):
    """95% CI for the mean, resampling segments rather than frames."""
    if values.max() == values.min():
        return float(values.mean()), float(values.mean())
    sums, counts = _segment_sums(values, segments)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def image_ci(values, n_boot=10000, seed=0, alpha=0.05):
    """95% CI resampling frames, for comparison with cluster_ci."""
    if values.max() == values.min():
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[picks].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_cluster_ci(a, b, segments, n_boot=10000, seed=0, alpha=0.05):
    """95% CI for mean(a) - mean(b) on the same frames."""
    diff = a - b
    if diff.max() == diff.min():
        return float(diff.mean()), float(diff.mean())
    return cluster_ci(diff, segments, n_boot, seed, alpha)


def half_width(lo, hi):
    return (hi - lo) / 2

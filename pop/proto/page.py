"""Page-limit basis sizing, per node (H1).

The owner's N-L2: the Page limit dictates how many vectors are needed to span one
equation's trajectory space -- the basis *per node* -- pruning degrees of freedom that
add no new mutual information.

Mechanism: for a node's empirical count matrix, compute singular values and keep only
directions above the Marchenko-Pastur upper edge for pure noise.  For a p x n noise
matrix with aspect ratio q = p/n and noise scale sigma, sample-covariance eigenvalues
spread over [sigma^2 (1-sqrt(q))^2, sigma^2 (1+sqrt(q))^2].  Singular values above
sigma*(1+sqrt(q)) are signal.

The noise scale is estimated from the bulk (median singular value / MP median).
"""

import numpy as np


def mp_upper_edge(p, n, sigma=1.0):
    """Upper edge of the Marchenko-Pastur bulk for the singular values of a
    p x n noise matrix with entry std sigma: sigma*sqrt(n)*(1+sqrt(p/n))."""
    q = p / n
    return sigma * np.sqrt(n) * (1.0 + np.sqrt(q))


def _mp_median(q, n_grid=20001):
    """Median of the MP bulk singular-value distribution (numeric)."""
    lo, hi = (1.0 - np.sqrt(q)) ** 2, (1.0 + np.sqrt(q)) ** 2
    xs = np.linspace(lo, hi, n_grid)
    # MP density for eigenvalues: sqrt((hi-x)(x-lo)) / (2 pi q x)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.sqrt(np.maximum(0.0, (hi - xs) * (xs - lo))) / (2 * np.pi * q * xs)
    dens[~np.isfinite(dens)] = 0.0
    cdf = np.cumsum(dens)
    cdf /= cdf[-1]
    med_eig = xs[np.searchsorted(cdf, 0.5)]
    return np.sqrt(med_eig)


def page_rank(counts, min_count=8):
    """Select the node's basis rank by the Page (MP-edge) criterion.

    counts: 2D array, rows = basis configurations, cols = outcomes.
    Returns (rank, singular_values, edge).  Rank 0 means "no signal above noise"
    when data are too scarce -- the node stays dormant (blanket closed).
    """
    counts = np.asarray(counts, dtype=float)
    p, n = counts.shape
    total = counts.sum()
    if total < min_count:
        return 0, np.array([]), float("nan")
    # Center columns (remove the mean outcome profile = the bulk direction).
    centered = counts - counts.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, compute_uv=False)
    if sv.size == 0 or sv[0] <= 0:
        return 0, sv, float("nan")
    q = p / n
    med = _mp_median(q)
    # median(sv) ~= sigma*sqrt(n)*med  =>  sigma = median(sv)/(sqrt(n)*med);
    # edge = sigma*sqrt(n)*(1+sqrt(q)) = median(sv)/med*(1+sqrt(q)).
    sigma_edge = np.median(sv) / med if med > 0 else sv[-1]
    edge = sigma_edge * (1.0 + np.sqrt(q))
    rank = int(np.sum(sv > edge))
    return rank, sv, edge

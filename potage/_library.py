"""Primitive library and selection history containers.

PrimitiveLibrary holds a feature matrix with per-column names and family labels.
All filter/dedup/subset operations return new libraries (immutable pattern),
and propagate to the optional validation split using train-derived decisions.

SelectionHistory records the ordered results of feature selection.
"""

import warnings

import numpy as np
from ._limits import DEDUP_HIERARCHICAL_THRESHOLD, DEDUP_BLOCK_SIZE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def warn_rank_features(X, names):
    """Warn if input features appear distribution-derived (rank/percentile).

    Rank and percentile features computed on the full dataset (train + test
    combined) leak ordering information from the test set into training.
    This function checks each column for the uniform-spacing signature of
    rank transforms: all n values are distinct and their sorted differences
    have near-zero variance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    names : list of str

    Returns
    -------
    suspect : list of str
        Names of features that look distribution-derived.
    """
    n = X.shape[0]
    if n < 30:
        return []

    suspect = []
    for j in range(X.shape[1]):
        col = X[:, j]
        n_unique = len(np.unique(col))
        # Rank features have (nearly) all distinct values
        if n_unique < n * 0.90:
            continue
        sorted_vals = np.sort(col)
        diffs = np.diff(sorted_vals)
        mean_diff = np.mean(diffs)
        if mean_diff < 1e-12:
            continue
        # Coefficient of variation of spacing: rank → ~0, natural data → >0.3
        cv = np.std(diffs) / mean_diff
        if cv < 0.05:
            suspect.append(names[j])

    if suspect:
        shown = ', '.join(suspect[:5])
        if len(suspect) > 5:
            shown += f', ... ({len(suspect)} total)'
        warnings.warn(
            f"Features appear distribution-derived (rank/percentile): "
            f"{shown}. If these were computed on train+test jointly, "
            f"this causes data leakage. Compute rank/percentile per fold "
            f"or use soup's built-in rank features instead.",
            UserWarning,
            stacklevel=3,
        )
    return suspect


def _normalize_zscore(X, n_sub=5000, seed=42):
    """Z-score normalize columns using a subsample.

    Returns (X_norm, means, stds, idx) where idx is the subsample indices.
    Constant columns get std=1.0 to avoid division by zero.
    """
    n = X.shape[0]
    n_sub = min(n_sub, n)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, n_sub, replace=False) if n > n_sub else np.arange(n)
    X_sub = X[idx]
    means = X_sub.mean(axis=0)
    stds = X_sub.std(axis=0)
    stds[stds < 1e-10] = 1.0
    X_norm = (X_sub - means) / stds
    return X_norm, means, stds, idx


class PrimitiveLibrary:
    """Feature matrix container with names, families, and optional val split.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training feature matrix.
    names : list of str
        Per-feature names.
    families : list of str
        Per-feature family labels.
    X_val : array-like of shape (n_val, n_features), optional
        Validation feature matrix. Filter/dedup/subset operations
        propagate to val using train-derived decisions.
    """

    def __init__(self, X, names, families, X_val=None, complexities=None):
        self.X = np.asarray(X, dtype=np.float64)
        self.names = list(names)
        self.families = list(families)
        if self.X.shape[1] != len(self.names) or len(self.names) != len(self.families):
            raise ValueError(
                f"Shape mismatch: X has {self.X.shape[1]} columns, "
                f"but got {len(self.names)} names and {len(self.families)} families"
            )
        self.X_val = np.asarray(X_val, dtype=np.float64) if X_val is not None else None
        if self.X_val is not None and self.X_val.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"X_val has {self.X_val.shape[1]} columns, expected {self.X.shape[1]}"
            )
        self.complexities = list(complexities) if complexities is not None else None
        if self.complexities is not None and len(self.complexities) != len(self.names):
            raise ValueError(
                f"complexities has {len(self.complexities)} entries, "
                f"expected {len(self.names)}"
            )
        self._corr_cache = None

    @property
    def n_features(self):
        return self.X.shape[1]

    @property
    def n_samples(self):
        return self.X.shape[0]

    def __repr__(self):
        fams = sorted(set(self.families))
        val_str = f", n_val={self.X_val.shape[0]}" if self.X_val is not None else ""
        return (f"PrimitiveLibrary({self.n_samples} samples, {self.n_features} features, "
                f"{len(fams)} families{val_str})")

    def _get_corr(self, seed=42):
        """Cached absolute correlation matrix (subsampled for speed)."""
        if self._corr_cache is not None:
            return self._corr_cache
        X_norm, _, _, _ = _normalize_zscore(self.X, seed=seed)
        n_sub = X_norm.shape[0]
        corr = np.abs(X_norm.T @ X_norm / n_sub)
        np.fill_diagonal(corr, 1.0)
        self._corr_cache = corr
        return corr

    def filter_constant(self, threshold=0.005):
        """Remove near-zero-variance columns (based on train stats).

        Returns a new PrimitiveLibrary.
        """
        stds = self.X.std(axis=0)
        keep = stds > threshold
        keep_idx = np.where(keep)[0]
        names_arr = np.array(self.names, dtype=object)
        fams_arr = np.array(self.families, dtype=object)
        return PrimitiveLibrary(
            self.X[:, keep],
            names_arr[keep_idx].tolist(),
            fams_arr[keep_idx].tolist(),
            X_val=self.X_val[:, keep] if self.X_val is not None else None,
            complexities=[self.complexities[i] for i in keep_idx] if self.complexities else None,
        )

    def dedup_correlated(self, threshold=0.95, seed=42, method='auto',
                         block_size=DEDUP_BLOCK_SIZE, representative='variance', y=None):
        """Correlation-based dedup -- keep best representative per component.

        For small libraries (d < 3000), uses hierarchical clustering
        (complete-linkage).  For large libraries, uses block-wise
        correlation with union-find to avoid O(d^2) memory.

        Clustering uses train data statistics.  Val columns (if present)
        follow the same keep decisions.

        Parameters
        ----------
        threshold : float
            |corr| above which features are considered duplicates.
        seed : int
            Random seed for subsampling.
        method : str
            'auto' (default), 'hierarchical', or 'union-find'.
            'auto' picks hierarchical for d < 3000, union-find otherwise.
        block_size : int
            Block size for union-find method.  Controls peak memory:
            O(block_size * d) instead of O(d^2).
        representative : str
            How to pick the kept feature per component:
            'variance' (default) -- highest std
            'complexity' -- lowest complexity (requires complexities)
            'correlation' -- highest |corr| with y (requires y)
        y : array-like, optional
            Target vector, required when representative='correlation'.

        Returns
        -------
        PrimitiveLibrary with deduplicated columns.
        """
        if method not in ('auto', 'hierarchical', 'union-find'):
            raise ValueError(f"Unknown dedup method: {method!r}")
        if representative not in ('variance', 'complexity', 'correlation'):
            raise ValueError(f"Unknown representative policy: {representative!r}")
        if representative == 'complexity' and self.complexities is None:
            raise ValueError("complexity representatives require complexities")
        if representative == 'correlation' and y is None:
            raise ValueError("correlation representatives require y")
        if self.n_features <= 1:
            return PrimitiveLibrary(
                self.X.copy(), list(self.names), list(self.families),
                X_val=self.X_val.copy() if self.X_val is not None else None,
                complexities=self.complexities,
            )

        d = self.n_features

        # Method selection
        if method == 'auto':
            use_uf = d >= DEDUP_HIERARCHICAL_THRESHOLD
        elif method == 'union-find':
            use_uf = True
        else:
            use_uf = False

        stds = self.X.std(axis=0)

        if use_uf:
            keep = self._dedup_union_find(threshold, seed, block_size,
                                          representative, y, stds)
        else:
            keep = self._dedup_hierarchical(threshold, seed, stds, representative, y)

        return PrimitiveLibrary(
            self.X[:, keep],
            [self.names[i] for i in keep],
            [self.families[i] for i in keep],
            X_val=self.X_val[:, keep] if self.X_val is not None else None,
            complexities=[self.complexities[i] for i in keep] if self.complexities else None,
        )

    def _dedup_hierarchical(self, threshold, seed, stds, representative, y):
        """Original hierarchical clustering dedup (complete-linkage)."""
        corr = self._get_corr(seed=seed)
        dist = np.clip(1.0 - corr, 0, 2)

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='complete')
        labels = fcluster(Z, t=1.0 - threshold, criterion='distance') - 1

        keep = []
        for c in range(labels.max() + 1):
            members = np.where(labels == c)[0]
            best = self._pick_representative(members, representative, y, stds)
            keep.append(best)
        return sorted(keep)

    def _dedup_union_find(self, threshold, seed, block_size,
                          representative, y, stds):
        """Block-wise correlation + union-find dedup.

        Memory: O(block_size * d) peak instead of O(d^2).
        """
        X_norm, _, _, _ = _normalize_zscore(self.X, seed=seed)
        n_sub, d = X_norm.shape

        # Shuffle column order to break systematic ordering bias
        rng = np.random.RandomState(seed)
        perm = rng.permutation(d)
        X_shuf = X_norm[:, perm]

        # Sign-hash pre-filter: find near-exact duplicates in O(d)
        # Hash sign patterns on a small subsample
        n_hash = min(256, n_sub)
        hash_idx = rng.choice(n_sub, n_hash, replace=False) if n_sub > n_hash else np.arange(n_sub)
        signs = (X_shuf[hash_idx] > 0).astype(np.uint8)
        # Convert each column's sign pattern to bytes for hashing
        sign_hashes = {}
        for j in range(d):
            h = signs[:, j].tobytes()
            if h in sign_hashes:
                sign_hashes[h].append(j)
            else:
                sign_hashes[h] = [j]

        # Union-Find with path compression + union by rank
        parent = list(range(d))
        rank_ = [0] * d

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank_[ra] < rank_[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank_[ra] == rank_[rb]:
                rank_[ra] += 1

        # Phase 1: Merge sign-hash collisions via exact correlation check
        for bucket in sign_hashes.values():
            if len(bucket) < 2:
                continue
            # Check pairwise within small buckets
            if len(bucket) <= 50:
                cols = X_shuf[:, bucket]
                corr_b = np.abs(cols.T @ cols / n_sub)
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        if corr_b[i, j] > threshold:
                            union(bucket[i], bucket[j])

        # Phase 2: Block-wise correlation scan
        for b_start in range(0, d, block_size):
            b_end = min(b_start + block_size, d)
            block = X_shuf[:, b_start:b_end]

            # Within-block correlations
            corr_bb = np.abs(block.T @ block / n_sub)
            rows, cols_ = np.where(np.triu(corr_bb, k=1) > threshold)
            for r, c in zip(rows, cols_):
                union(b_start + r, b_start + c)

            # Cross-block correlations (block vs all previous blocks)
            if b_start > 0:
                prev = X_shuf[:, :b_start]
                corr_bp = np.abs(prev.T @ block / n_sub)  # (b_start, block_width)
                rows, cols_ = np.where(corr_bp > threshold)
                for r, c in zip(rows, cols_):
                    union(r, b_start + c)

        # Each union-find root uniquely identifies a component.
        components = {}
        for j in range(d):
            components.setdefault(find(j), []).append(perm[j])

        # Post-check: verify every member correlates >= threshold with representative
        # This mitigates single-linkage chaining
        keep = []
        for members in components.values():
            if len(members) == 1:
                keep.append(members[0])
                continue

            rep = self._pick_representative(members, representative, y, stds)

            # Post-check: only keep rep, eject members that don't
            # correlate with rep above threshold (they become their own reps)
            if len(members) <= 20:
                # For small components, do exact check
                rep_col = X_norm[:, rep]
                for m in members:
                    if m == rep:
                        continue
                    c = abs(float(rep_col @ X_norm[:, m] / n_sub))
                    if c < threshold:
                        keep.append(m)  # eject: doesn't correlate with rep
                keep.append(rep)
            else:
                # For large components, just keep the representative
                # (post-check too expensive, and large components at 0.95
                # threshold are rare)
                keep.append(rep)

        return sorted(keep)

    def _pick_representative(self, members, method, y, stds):
        """Pick the best representative from a component."""
        members_arr = np.array(members)
        if method == 'complexity' and self.complexities is not None:
            complexities = np.array([self.complexities[m] for m in members])
            # Lowest complexity wins; break ties by highest variance
            min_c = complexities.min()
            ties = members_arr[complexities == min_c]
            return int(ties[np.argmax(stds[ties])])
        elif method == 'correlation' and y is not None:
            y = np.asarray(y, dtype=np.float64)
            y_c = y - y.mean()
            y_std = y.std()
            if y_std < 1e-10:
                return int(members_arr[np.argmax(stds[members_arr])])
            n = len(y)
            X_m = self.X[:, members_arr]
            x_c = X_m - X_m.mean(axis=0)
            x_stds = X_m.std(axis=0)
            x_stds[x_stds < 1e-10] = 1.0
            corrs = np.abs(x_c.T @ y_c) / (x_stds * y_std * n)
            return int(members_arr[np.argmax(corrs)])
        else:
            # Default: highest variance
            return int(members_arr[np.argmax(stds[members_arr])])

    def correlation_matrix(self, seed=42):
        """Pairwise |corr| matrix of current features (cached)."""
        return self._get_corr(seed=seed)

    def family_correlation(self):
        """Mean |corr| between each pair of families.

        Returns
        -------
        result : dict mapping (fam_a, fam_b) -> mean |corr|
        unique_families : list of str
        """
        corr = self._get_corr()
        unique_fams = sorted(set(self.families))
        fam_idx = {f: [i for i, ff in enumerate(self.families) if ff == f]
                   for f in unique_fams}
        result = {}
        for fa in unique_fams:
            for fb in unique_fams:
                if fa > fb:
                    continue
                ia, ib = fam_idx[fa], fam_idx[fb]
                block = corr[np.ix_(ia, ib)]
                if fa == fb:
                    mask = ~np.eye(len(ia), len(ib), dtype=bool)
                    if mask.sum() > 0:
                        result[(fa, fb)] = float(block[mask].mean())
                    else:
                        result[(fa, fb)] = 1.0
                else:
                    result[(fa, fb)] = float(block.mean())
        return result, unique_fams

    def filter_complexity(self, max_complexity):
        """Return a new library keeping only features with complexity <= max.

        Parameters
        ----------
        max_complexity : int
            Maximum allowed complexity.  Features without complexity
            metadata are always kept.
        """
        if self.complexities is None:
            return self  # no metadata, keep everything
        keep = [i for i, c in enumerate(self.complexities) if c <= max_complexity]
        return PrimitiveLibrary(
            self.X[:, keep],
            [self.names[i] for i in keep],
            [self.families[i] for i in keep],
            X_val=self.X_val[:, keep] if self.X_val is not None else None,
            complexities=[self.complexities[i] for i in keep],
        )

    def subset_families(self, families):
        """Return a new library containing only features from the given families."""
        fam_set = set(families)
        keep = [i for i, f in enumerate(self.families) if f in fam_set]
        return PrimitiveLibrary(
            self.X[:, keep],
            [self.names[i] for i in keep],
            [self.families[i] for i in keep],
            X_val=self.X_val[:, keep] if self.X_val is not None else None,
            complexities=[self.complexities[i] for i in keep] if self.complexities else None,
        )

    def filter_families(self, exclude):
        """Return a new library excluding features from the given families.

        Parameters
        ----------
        exclude : str or set of str
            Family name(s) to remove.
        """
        if isinstance(exclude, str):
            exclude = {exclude}
        else:
            exclude = set(exclude)
        keep = [i for i, f in enumerate(self.families) if f not in exclude]
        return PrimitiveLibrary(
            self.X[:, keep],
            [self.names[i] for i in keep],
            [self.families[i] for i in keep],
            X_val=self.X_val[:, keep] if self.X_val is not None else None,
            complexities=[self.complexities[i] for i in keep] if self.complexities else None,
        )


class SelectionHistory:
    """Ordered record of feature selection steps.

    ``score_kind`` identifies R², mRMR, or a classification union;
    ``score_split`` identifies train, validation, or fold-validation scoring.
    Step keys retain legacy R² names for compatibility. OMP records actual
    R² gains in marginal_r2 and its ranking correlation in selection_score.
    """

    def __init__(self, score_kind='r2', score_split='train'):
        self.steps = []
        self.score_kind = score_kind
        self.score_split = score_split

    def append(self, name, family, marginal_r2, cumulative_r2,
               fold_r2_std=None, selection_score=None):
        step = {
            'name': name,
            'family': family,
            'marginal_r2': float(marginal_r2),
            'cumulative_r2': float(cumulative_r2),
        }
        if selection_score is not None:
            step['selection_score'] = float(selection_score)
        if fold_r2_std is not None:
            step['fold_r2_std'] = float(fold_r2_std)
        self.steps.append(step)

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        if not self.steps:
            return "SelectionHistory(empty)"
        final_r2 = self.steps[-1]['cumulative_r2']
        return f"SelectionHistory({len(self.steps)} steps, {self.score_kind}={final_r2:.4f})"

    def cumulative_r2(self):
        return [s['cumulative_r2'] for s in self.steps]

    def marginal_r2(self):
        return [s['marginal_r2'] for s in self.steps]

    def selected_names(self):
        return [s['name'] for s in self.steps]

    def selected_families(self):
        return [s['family'] for s in self.steps]

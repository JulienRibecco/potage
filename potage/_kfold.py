"""K-fold averaged feature selection helpers.

Extracted from _pipe.py — these are standalone functions with no
dependency on SoupPipe internals.
"""

import warnings

import numpy as np


def _align_val_cols(train_names, val_names, val_cols, n_val):
    """Align validation columns to match train column order by name.

    One-hot encoding can produce different columns for train vs val
    (different unique values). This matches val to train by name,
    filling missing columns with zeros.

    Returns list of ndarray parallel to train_names.
    """
    val_lookup = dict(zip(val_names, val_cols))
    aligned = []
    for name in train_names:
        if name in val_lookup:
            aligned.append(val_lookup[name])
        else:
            aligned.append(np.zeros(n_val))
    return aligned


def _subset_by_names(cols, pnames, selected_names, n_new):
    """Extract selected columns by name from generated candidates.

    Parameters
    ----------
    cols : list of ndarray
        Generated candidate columns.
    pnames : list of str
        Names parallel to cols.
    selected_names : list of str
        Names of features to extract (from training selection).
    n_new : int
        Number of rows in the new data.

    Returns
    -------
    ndarray of shape (n_new, len(selected_names))
    """
    name_to_col = {}
    for name, col in zip(pnames, cols):
        if name not in name_to_col:
            name_to_col[name] = col

    result_cols = []
    missing = []
    for name in selected_names:
        if name in name_to_col:
            result_cols.append(name_to_col[name])
        else:
            missing.append(name)
            result_cols.append(np.zeros(n_new))

    if missing:
        warnings.warn(
            f"{len(missing)} selected feature(s) not found during transform "
            f"(zero-filled): {missing[:5]}{'...' if len(missing) > 5 else ''}",
            RuntimeWarning,
        )

    if not result_cols:
        return np.empty((n_new, 0))
    return np.column_stack(result_cols)


# ---------------------------------------------------------------------------
# K-fold averaged selection
# ---------------------------------------------------------------------------

def _make_folds(n_samples, K, seed):
    """Create K fold index arrays from a random permutation."""
    if not isinstance(K, (int, np.integer)) or isinstance(K, bool) or not 2 <= K <= n_samples // 2:
        raise ValueError("fold_eval_count must be an integer >= 2 with at least two rows per fold")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    fold_sizes = [n_samples // K + (1 if i < n_samples % K else 0)
                  for i in range(K)]
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(perm[start:start + size])
        start += size
    return folds


from ._limits import GRAM_FREE_THRESHOLD as _GRAM_FREE_THRESHOLD

_FOLD_VARIANCE_THRESHOLD = 0.05  # R² std across folds


def _warn_fold_variance(history):
    """Warn if any selected feature causes high fold R² variance.

    High variance means the feature's contribution is unstable across
    folds — it may overfit or cause prediction blow-ups on new data.
    """
    if history.score_kind != "r2":
        return
    unstable = []
    for i, step in enumerate(history.steps):
        std = step.get('fold_r2_std')
        if std is not None and std > _FOLD_VARIANCE_THRESHOLD:
            unstable.append((i + 1, step['name'], std))

    if unstable:
        lines = [f"  #{rank} {name} (fold R² std={std:.3f})"
                 for rank, name, std in unstable[:5]]
        n_extra = len(unstable) - 5
        msg = (f"high fold variance in {len(unstable)} selected feature(s) "
               f"(R² std > {_FOLD_VARIANCE_THRESHOLD}):\n"
               + "\n".join(lines))
        if n_extra > 0:
            msg += f"\n  ... and {n_extra} more"
        msg += ("\nConsider excluding their families via "
                "exclude_families= or reducing n_select.")
        warnings.warn(msg, stacklevel=4)


def _kfold_select(X, names, families, y, n_select, method, K, seed,
                  alpha=1.0, _gram_free='auto'):
    """Unified K-fold averaged feature selection.

    Greedy uses validation R² to rank candidates. OMP aggregates training
    residual correlations and reports held-out fold R². Correlation selection
    averages relevance/redundancy scores; these are not model R² values.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Filtered/deduped candidate matrix (full training data).
    names, families : lists parallel to columns of X.
    y : ndarray (n_samples,)
    n_select : int
    method : 'greedy', 'omp', or 'correlation'
    K : int — number of folds
    seed : int
    alpha : float — ridge alpha for greedy
    _gram_free : str or bool
        When 'auto', switches greedy/omp to correlation method for
        d > threshold to avoid K×d² Gram matrices.
    """
    n_features = X.shape[1]

    # Resolve gram-free: auto-switch to correlation for large d
    if _gram_free == 'auto':
        gram_free = n_features >= _GRAM_FREE_THRESHOLD
    else:
        gram_free = bool(_gram_free)

    if gram_free and method in ('greedy', 'omp'):
        warnings.warn(
            f"_kfold_select: auto-switching from {method!r} to 'correlation' "
            f"for d={n_features} (>{_GRAM_FREE_THRESHOLD}) to avoid "
            f"K×d² Gram matrices",
            RuntimeWarning,
            stacklevel=2,
        )
        method = 'correlation'

    fold_indices = _make_folds(X.shape[0], K, seed)
    if method == 'greedy':
        return _kfold_greedy(X, names, families, y, n_select,
                             fold_indices, alpha)
    elif method == 'omp':
        return _kfold_omp(X, names, families, y, n_select, fold_indices)
    elif method == 'correlation':
        return _kfold_correlation(X, names, families, y, n_select,
                                  fold_indices)
    else:
        raise ValueError(f"Unknown method for kfold select: {method!r}")


def _kfold_greedy(X, names, families, y, n_select, fold_indices, alpha=1.0):
    """K-fold greedy forward selection with incremental Gram inverse.

    Maintains K independent fold states; at each step the val R² is
    averaged across folds to choose the next feature.
    """
    from ._library import SelectionHistory

    n_samples, n_features = X.shape
    K = len(fold_indices)

    # Build per-fold precomputed state
    folds = []
    for val_idx in fold_indices:
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_idx] = False

        X_tr = X[train_mask]
        y_tr = y[train_mask]
        X_val = X[val_idx]
        y_val = y[val_idx]

        n_tr = X_tr.shape[0]
        n_val = len(val_idx)
        mu = X_tr.mean(axis=0)
        y_mean = float(y_tr.mean())
        G = X_tr.T @ X_tr
        G_c = G - n_tr * np.outer(mu, mu)
        Xy = X_tr.T @ y_tr
        Xy_c = Xy - n_tr * mu * y_mean
        ss_tot_val = float(np.var(y_val) * n_val)

        folds.append({
            'X_val': X_val, 'y_val': y_val,
            'mu': mu, 'y_mean': y_mean,
            'G_c': G_c, 'Xy_c': Xy_c,
            'G_c_diag_reg': np.diag(G_c).copy() + alpha,
            'n_val': n_val,
            'ss_tot_val': ss_tot_val,
            'A_inv': np.empty((0, 0)),
            'u': np.empty(0),
        })

    active_idx = []
    remaining = np.ones(n_features, dtype=bool)
    history = SelectionHistory('r2', 'fold_validation')
    prev_r2 = 0.0

    for step in range(n_select):
        rem_idx = np.where(remaining)[0]
        n_rem = len(rem_idx)
        if n_rem == 0:
            break

        k = len(active_idx)
        active_arr = np.array(active_idx, dtype=np.intp)

        # Average val R² across folds
        r2_sum = np.zeros(n_rem)
        r2_per_fold = []  # collect per-fold R² vectors for variance
        any_valid = np.zeros(n_rem, dtype=bool)

        for fold in folds:
            G_c = fold['G_c']
            Xy_c = fold['Xy_c']
            G_c_diag_reg = fold['G_c_diag_reg']
            A_inv = fold['A_inv']
            u = fold['u']

            G_cross = (G_c[np.ix_(active_arr, rem_idx)]
                       if k > 0 else np.empty((0, n_rem)))
            V = A_inv @ G_cross
            schur = G_c_diag_reg[rem_idx] - (G_cross * V).sum(axis=0)
            delta = Xy_c[rem_idx] - (G_cross.T @ u if k > 0 else 0.0)

            valid = schur > 1e-12
            schur_safe = np.where(valid, schur, 1.0)
            cl = delta / schur_safe

            # Batch val predictions
            X_val = fold['X_val']
            y_val = fold['y_val']
            mu = fold['mu']
            y_mean = fold['y_mean']
            n_val = fold['n_val']
            ss_tot = fold['ss_tot_val']

            if k > 0:
                X_va = X_val[:, active_arr]
                intercept = y_mean - mu[active_arr] @ u
                pred_base = X_va @ u + intercept
                mu_corr = mu[active_arr] @ V - mu[rem_idx]
                corrections = (X_val[:, rem_idx] - X_va @ V
                               + mu_corr[None, :])
            else:
                pred_base = np.full(n_val, y_mean)
                corrections = X_val[:, rem_idx] - mu[rem_idx][None, :]

            pred_all = pred_base[:, None] + cl[None, :] * corrections
            ss_res = ((y_val[:, None] - pred_all) ** 2).sum(axis=0)
            ss_res = np.where(valid, ss_res, ss_tot * 2.0)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-15)

            r2_sum += r2
            r2_per_fold.append(r2.copy())
            any_valid |= valid

        r2_avg = r2_sum / K
        r2_avg[~any_valid] = -np.inf

        best_local = int(np.argmax(r2_avg))
        best_r2 = float(r2_avg[best_local])
        best_j = int(rem_idx[best_local])

        if not any_valid[best_local]:
            break

        marginal = best_r2 - prev_r2
        # Stop if average val R² decreased (overshoot)
        if step > 0 and marginal < -1e-8:
            break

        # Update all K fold states
        for fold in folds:
            G_c = fold['G_c']
            G_c_diag_reg = fold['G_c_diag_reg']
            A_inv = fold['A_inv']

            if k == 0:
                fold['A_inv'] = np.array([[1.0 / G_c_diag_reg[best_j]]])
            else:
                g = G_c[active_arr, best_j]
                v = A_inv @ g
                s_inv = 1.0 / (G_c_diag_reg[best_j] - g @ v)
                k1 = k + 1
                A_new = np.empty((k1, k1))
                A_new[:k, :k] = A_inv + s_inv * np.outer(v, v)
                A_new[:k, k] = -s_inv * v
                A_new[k, :k] = -s_inv * v
                A_new[k, k] = s_inv
                fold['A_inv'] = A_new

            new_active = np.array(active_idx + [best_j], dtype=np.intp)
            fold['u'] = fold['A_inv'] @ fold['Xy_c'][new_active]

        active_idx.append(best_j)
        remaining[best_j] = False
        # Fold R² std for the winner
        winner_r2s = [rpf[best_local] for rpf in r2_per_fold]
        fold_std = float(np.std(winner_r2s))
        history.append(names[best_j], families[best_j], marginal, best_r2,
                       fold_r2_std=fold_std)
        prev_r2 = best_r2

    # Warn about high fold variance in selected features
    _warn_fold_variance(history)

    return history


def _kfold_omp(X, names, families, y, n_select, fold_indices):
    """K-fold averaged OMP selection.

    K independent residuals; at each step the average |corr with residual|
    across training folds picks the next feature. Reported R² is computed
    on the excluded validation rows, not on the fitting rows.
    """
    from ._library import SelectionHistory
    from sklearn.metrics import r2_score

    n_samples, n_features = X.shape
    K = len(fold_indices)

    folds = []
    for val_idx in fold_indices:
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_idx] = False

        X_tr = X[train_mask]
        y_tr = y[train_mask]
        stds = X_tr.std(axis=0)
        valid = stds > 1e-10
        stds_safe = stds.copy()
        stds_safe[~valid] = 1.0

        X_tr_mean = X_tr.mean(axis=0)
        folds.append({
            'X_tr': X_tr, 'y_tr': y_tr,
            'X_c': X_tr - X_tr_mean,
            'stds': stds_safe, 'valid': valid,
            'G': (X_tr - X_tr_mean).T @ (X_tr - X_tr_mean),
            'Xy': (X_tr - X_tr_mean).T @ (y_tr - y_tr.mean()),
            'X_val': X[val_idx], 'y_val': y[val_idx],
            'mu': X_tr_mean, 'y_mean': y_tr.mean(),
            'residual': y_tr - y_tr.mean(),
        })

    selected_idx = []
    history = SelectionHistory('r2', 'fold_validation')

    for rank in range(n_select):
        # Average |corr with residual| across folds
        corr_sum = np.zeros(n_features)
        n_active_folds = 0

        for fold in folds:
            r_std = fold['residual'].std()
            if r_std < 1e-10:
                continue
            n_tr = fold['X_tr'].shape[0]
            r_c = fold['residual'] - fold['residual'].mean()
            corrs = np.abs(fold['X_c'].T @ r_c) / (
                fold['stds'] * r_std * n_tr)
            corrs[~fold['valid']] = 0.0
            corr_sum += corrs
            n_active_folds += 1

        if n_active_folds == 0:
            break
        corr_avg = corr_sum / n_active_folds

        # Mask selected
        if selected_idx:
            corr_avg[selected_idx] = -1.0

        best_j = int(np.argmax(corr_avg))
        if corr_avg[best_j] < 0:
            break

        selected_idx.append(best_j)

        # Refit OLS on each fold, update residuals, compute avg R²
        sel_arr = np.array(selected_idx)
        n_sel = len(selected_idx)
        fold_r2s = []

        for fold in folds:
            G_sel = fold['G'][np.ix_(sel_arr, sel_arr)] + 1e-6 * np.eye(n_sel)
            Xy_sel = fold['Xy'][sel_arr]
            coef = np.linalg.lstsq(G_sel, Xy_sel, rcond=None)[0]
            y_hat = fold['X_c'][:, sel_arr] @ coef + fold['y_mean']
            fold['residual'] = fold['y_tr'] - y_hat
            val_hat = ((fold['X_val'][:, sel_arr] - fold['mu'][sel_arr])
                       @ coef + fold['y_mean'])
            r2 = r2_score(fold['y_val'], val_hat)
            fold_r2s.append(r2)

        r2_avg = np.mean(fold_r2s)
        r2_std = float(np.std(fold_r2s))
        history.append(names[best_j], families[best_j],
                       r2_avg - (history.steps[-1]['cumulative_r2'] if history.steps else 0.0),
                       r2_avg, fold_r2_std=r2_std,
                       selection_score=float(corr_avg[best_j]))

    # Warn about high fold variance in selected features
    _warn_fold_variance(history)

    return history


def _kfold_correlation(X, names, families, y, n_select, fold_indices,
                       diversity=0.5):
    """K-fold averaged mRMR selection.

    Averages |corr(feature, y)| and the inter-feature Gram across folds.
    """
    from ._library import SelectionHistory

    n_samples, n_features = X.shape
    K = len(fold_indices)

    corr_sum = np.zeros(n_features)
    gram_sum = np.zeros((n_features, n_features)) if diversity > 0 else None

    for val_idx in fold_indices:
        X_v = X[val_idx]
        y_v = y[val_idx]
        n_v = len(val_idx)

        x_means = X_v.mean(axis=0)
        x_stds = X_v.std(axis=0)
        y_std = y_v.std()
        valid = x_stds > 1e-10
        x_stds_s = x_stds.copy()
        x_stds_s[~valid] = 1.0

        X_c = X_v - x_means
        y_c = y_v - y_v.mean()
        corrs = np.abs(X_c.T @ y_c) / (x_stds_s * max(y_std, 1e-15) * n_v)
        corrs[~valid] = 0.0
        corr_sum += corrs

        if diversity > 0:
            stds = x_stds.copy()
            stds[stds < 1e-10] = 1.0
            X_norm = (X_v - x_means) / stds
            gram_sum += np.abs(X_norm.T @ X_norm / n_v)

    corr_avg = corr_sum / K
    gram_avg = gram_sum / K if diversity > 0 else None

    selected = []
    remaining = set(range(n_features))
    history = SelectionHistory('mrmr', 'fold_validation')

    for rank in range(min(n_select, n_features)):
        rem_arr = np.array(sorted(remaining))
        rem_corrs = corr_avg[rem_arr]
        nonzero = rem_corrs > 0
        if not nonzero.any():
            break

        if diversity > 0 and selected:
            sel_arr = np.array(selected)
            redundancy = gram_avg[np.ix_(rem_arr, sel_arr)].mean(axis=1)
            scores = rem_corrs - diversity * redundancy
        else:
            scores = rem_corrs.copy()

        scores[~nonzero] = -np.inf
        best_idx = int(np.argmax(scores))
        best_j = int(rem_arr[best_idx])

        if scores[best_idx] == -np.inf:
            break

        selected.append(best_j)
        remaining.discard(best_j)
        history.append(names[best_j], families[best_j],
                       float(corr_avg[best_j]), float(scores[best_idx]))

    return history

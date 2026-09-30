"""Feature selection algorithms for soup libraries.

Three complementary algorithms:
    greedy_forward_select  -- Ridge-based forward selection (default)
    correlation_select     -- mRMR-style univariate selection (small data)
    residual_select        -- OMP-style residual correlation (no hyperparams)
    per_class_select       -- One-vs-all wrapper for classification targets
"""

import re
import time
import numpy as np

from ._library import SelectionHistory

from ._limits import GRAM_FREE_THRESHOLD as _GRAM_FREE_THRESHOLD


def _find_component_names(feature_name, all_names):
    """Find library feature names that appear as components in a compound name.

    Uses word-boundary matching so 'mean_N' won't false-match inside
    'mean_NValence', but 'entropy_MeltingT' will match inside
    'entropy_MeltingT/(1+|std_Electronegativity|)'.

    Returns list of matching parent names (longest first).
    """
    components = []
    for name in all_names:
        if name == feature_name or len(name) >= len(feature_name):
            continue
        if name not in feature_name:
            continue  # fast substring pre-filter
        # Word boundary: not preceded/followed by [a-zA-Z0-9_.]
        pattern = r'(?<![a-zA-Z0-9_.])' + re.escape(name) + r'(?![a-zA-Z0-9_.])'
        if re.search(pattern, feature_name):
            components.append(name)
    # Longest first — more specific parents checked first
    components.sort(key=len, reverse=True)
    return components


def greedy_forward_select(library, y, max_steps=20, alpha=1.0,
                          y_val=None, r2_stop=0.9999, verbose=False,
                          lookahead=False, family_diversity=0.0,
                          parent_retain=0.0, min_delta_r2='auto',
                          _gram_free='auto'):
    """Ridge-based greedy forward selection via incremental inverse.

    At each step: batch-scores every remaining feature using a maintained
    inverse of the active Gram matrix. O(k^2 * d) per step instead of
    O(k^3 * d).

    When library.X_val is present and y_val is given, features are fitted
    on train data but scored on validation data.

    Parameters
    ----------
    library : PrimitiveLibrary
    y : array-like of shape (n_samples,)
        Training target.
    max_steps : int
        Maximum selection steps.
    alpha : float
        Ridge regularization parameter.
    y_val : array-like of shape (n_val,), optional
        Validation target. Enables val-controlled selection when
        library.X_val is also present.
    r2_stop : float
        Stop early when R^2 exceeds this threshold.
    verbose : bool
        Print progress per step.
    lookahead : bool
        If True, after picking the greedy best, also test top-N pairs
        to catch synergistic features weak alone but strong together.
    family_diversity : float
        Soft penalty for same-family redundancy (0=off, 1=full penalty).
        Penalizes candidates whose family already has many selected features.
    parent_retain : float
        When > 0, after selecting a compound feature, check if its parent
        (component) features add marginal R^2.  Parents whose marginal
        gain is at least ``parent_retain`` times the child's marginal gain
        are auto-injected into the active set.  Injected parents do NOT
        count toward ``max_steps``.  Set to 0 to disable (default).
    min_delta_r2 : float or 'auto'
        Minimum marginal R² improvement to accept a feature.  Stops
        selection when the best candidate improves R² by less than this.
        ``'auto'`` (default) sets the threshold to ``2 / n_samples`` —
        the expected R² of a single random feature, which prevents
        selecting features that contribute less than noise.
        Set to 0 to disable.

    Returns
    -------
    SelectionHistory
    """
    y = np.asarray(y, dtype=np.float64)
    X = library.X
    n_samples, n_features = X.shape
    if len(y) != n_samples:
        raise ValueError(f"y has {len(y)} samples, X has {n_samples}")

    total_var = np.var(y)
    if total_var < 1e-15:
        return SelectionHistory('r2', 'validation' if library.X_val is not None and y_val is not None else 'train')

    # Minimum marginal R² threshold (noise floor)
    if min_delta_r2 == 'auto':
        min_delta_r2 = 2.0 / n_samples
    min_delta_r2 = float(min_delta_r2)

    use_val = library.X_val is not None and y_val is not None

    # Precompute train data statistics
    mu = X.mean(axis=0)
    y_mean = y.mean()
    y_c = y - y_mean
    yy_c = y_c.dot(y_c)
    ss_tot_train = total_var * n_samples

    # Resolve gram-free mode: skip d×d Gram for large d
    if _gram_free == 'auto':
        gram_free = n_features >= _GRAM_FREE_THRESHOLD
    else:
        gram_free = bool(_gram_free)

    if gram_free:
        X_c = X - mu
        Xy_c = X_c.T @ y_c
        G_c_diag_reg = (X_c ** 2).sum(axis=0) + alpha
    else:
        G = X.T @ X
        G_c = G - n_samples * np.outer(mu, mu)
        Xy = X.T @ y
        Xy_c = Xy - n_samples * mu * y_mean
        G_c_diag_reg = np.diag(G_c).copy() + alpha

    if use_val:
        X_val = library.X_val
        y_val = np.asarray(y_val, dtype=np.float64)
        n_val = len(y_val)
        val_total_var = np.var(y_val)
        ss_tot_val = val_total_var * n_val

    active_idx = []
    remaining = np.ones(n_features, dtype=bool)
    history = SelectionHistory('r2', 'validation' if library.X_val is not None and y_val is not None else 'train')
    prev_r2 = 0.0
    t0 = time.time()

    # Parent redundancy pre-filter: mask out features that add < min_delta_r2
    # over their parent features.  A feature like cos(4pi*EN) that adds
    # nothing on top of the raw EN column is filtered here before the
    # selection loop even starts.
    if min_delta_r2 > 0 and yy_c > 1e-15:
        name_to_idx = {n: i for i, n in enumerate(library.names)}
        n_parent_filtered = 0
        for j in range(n_features):
            parents = _find_component_names(library.names[j], library.names)
            if not parents:
                continue  # stage-0 feature, no parent to check against
            pidx = [name_to_idx[p] for p in parents if p in name_to_idx]
            if not pidx:
                continue
            P = np.array(pidx, dtype=np.intp)
            # R²(y ~ parents) via centered OLS
            if gram_free:
                X_c_P = X_c[:, P]
                G_pp = X_c_P.T @ X_c_P + 1e-8 * np.eye(len(P))
            else:
                G_pp = G_c[np.ix_(P, P)] + 1e-8 * np.eye(len(P))
            Xy_p = Xy_c[P]
            try:
                w_p = np.linalg.solve(G_pp, Xy_p)
                r2_p = max(0.0, float(w_p @ Xy_p) / yy_c)
            except np.linalg.LinAlgError:
                continue
            # R²(y ~ parents + candidate)
            PJ = np.append(P, j)
            if gram_free:
                X_c_PJ = X_c[:, PJ]
                G_pj = X_c_PJ.T @ X_c_PJ + 1e-8 * np.eye(len(PJ))
            else:
                G_pj = G_c[np.ix_(PJ, PJ)] + 1e-8 * np.eye(len(PJ))
            Xy_pj = Xy_c[PJ]
            try:
                w_pj = np.linalg.solve(G_pj, Xy_pj)
                r2_pj = max(0.0, float(w_pj @ Xy_pj) / yy_c)
            except np.linalg.LinAlgError:
                continue
            if r2_pj - r2_p < min_delta_r2:
                remaining[j] = False
                n_parent_filtered += 1
        if verbose and n_parent_filtered > 0:
            print(f'    parent check: filtered {n_parent_filtered}/{n_features}'
                  f' features redundant with parents '
                  f'(min_delta_r2={min_delta_r2:.6f})')

    # Maintained incremental state
    A_inv = np.empty((0, 0))    # inverse of (G_c[active,active] + alpha*I)
    u = np.empty(0)             # A_inv @ Xy_c[active] = active coefs
    u_rhs = 0.0                 # u' @ Xy_c[active]
    u_norm_sq = 0.0             # ||u||^2

    for step in range(min(max_steps, n_features)):
        step_t0 = time.time()
        rem_idx = np.where(remaining)[0]
        n_rem = len(rem_idx)
        if n_rem == 0:
            break

        k = len(active_idx)
        active_arr = np.array(active_idx, dtype=np.intp)

        # Cross-terms between active set and remaining candidates
        if k > 0:
            if gram_free:
                G_cross = X_c[:, active_arr].T @ X_c[:, rem_idx]
            else:
                G_cross = G_c[np.ix_(active_arr, rem_idx)]
        else:
            G_cross = np.empty((0, n_rem))
        V = A_inv @ G_cross  # (k, n_rem)
        schur = G_c_diag_reg[rem_idx] - (G_cross * V).sum(axis=0)
        delta = Xy_c[rem_idx] - (G_cross.T @ u if k > 0 else 0.0)

        valid = schur > 1e-12
        schur_safe = np.where(valid, schur, 1.0)
        cl = delta / schur_safe  # coef_last per candidate

        if not use_val:
            # Batch train-only R^2 from Gram
            u_dot_V = u @ V if k > 0 else np.zeros(n_rem)
            V_nsq = (V * V).sum(axis=0) if k > 0 else np.zeros(n_rem)
            coef_nsq = u_norm_sq - 2.0 * cl * u_dot_V + cl * cl * (V_nsq + 1.0)
            ss_res = yy_c - (u_rhs + delta * cl) - alpha * coef_nsq
            ss_res = np.where(valid, ss_res, yy_c)
            r2_all = 1.0 - ss_res / ss_tot_train
        else:
            # Batch val predictions
            if k > 0:
                X_val_active = X_val[:, active_arr]
                intercept_base = y_mean - mu[active_arr] @ u
                pred_base = X_val_active @ u + intercept_base
                mu_corr = mu[active_arr] @ V - mu[rem_idx]
                corrections = X_val[:, rem_idx] - X_val_active @ V + mu_corr[None, :]
            else:
                pred_base = np.full(n_val, y_mean)
                corrections = X_val[:, rem_idx] - mu[rem_idx][None, :]

            pred_all = pred_base[:, None] + cl[None, :] * corrections
            ss_res = ((y_val[:, None] - pred_all) ** 2).sum(axis=0)
            ss_res = np.where(valid, ss_res, ss_tot_val * 2.0)
            r2_all = 1.0 - ss_res / ss_tot_val

        # Family diversity: penalize candidates from over-represented families
        r2_score = r2_all.copy()
        if family_diversity > 0 and k > 0:
            # Count how many active features share each family
            active_fam_count = {}
            for ai in active_idx:
                fam = library.families[ai]
                active_fam_count[fam] = active_fam_count.get(fam, 0) + 1
            for ri in range(n_rem):
                if not valid[ri]:
                    continue
                fam = library.families[rem_idx[ri]]
                cnt = active_fam_count.get(fam, 0)
                # Penalty proportional to fraction of selected from same family
                penalty = family_diversity * (cnt / k) * 0.01
                r2_score[ri] -= penalty

        best_local = int(np.argmax(r2_score))
        best_r2_raw = float(r2_all[best_local])
        best_j = int(rem_idx[best_local])

        # 2-step lookahead: check if a different first pick leads to
        # a better 2-step outcome (only if >1 remaining and step < max-1)
        if lookahead and n_rem > 1 and step < max_steps - 1:
            # Top-8 candidates by r2_score
            top_n = min(8, n_rem)
            top_locals = np.argsort(-r2_score)[:top_n]
            best_pair_r2 = best_r2_raw  # fallback: just greedy best alone

            for t1 in top_locals:
                if not valid[t1]:
                    continue
                j1 = int(rem_idx[t1])
                # Simulate adding j1: compute what the best 2nd pick would be
                # Use Schur complement for j1, then score remaining candidates
                if gram_free:
                    g1 = X_c[:, active_arr].T @ X_c[:, j1] if k > 0 else np.empty(0)
                else:
                    g1 = G_c[active_arr, j1] if k > 0 else np.empty(0)
                v1 = A_inv @ g1 if k > 0 else np.empty(0)
                s1 = G_c_diag_reg[j1] - (g1 @ v1 if k > 0 else 0.0)
                if s1 < 1e-12:
                    continue
                s1_inv = 1.0 / s1
                d1 = Xy_c[j1] - (g1 @ u if k > 0 else 0.0)
                cl1 = d1 / s1

                # Build extended inverse for {active + j1}
                k1 = k + 1
                if k == 0:
                    A1 = np.array([[s1_inv]])
                    u1 = np.array([cl1])
                else:
                    A1 = np.empty((k1, k1))
                    A1[:k, :k] = A_inv + s1_inv * np.outer(v1, v1)
                    A1[:k, k] = -s1_inv * v1
                    A1[k, :k] = -s1_inv * v1
                    A1[k, k] = s1_inv
                    a1_arr = np.array(active_idx + [j1], dtype=np.intp)
                    u1 = A1 @ Xy_c[a1_arr]

                # Score top-8 remaining (excluding j1)
                for t2 in top_locals:
                    if t2 == t1 or not valid[t2]:
                        continue
                    j2 = int(rem_idx[t2])
                    a1_arr = np.array(active_idx + [j1], dtype=np.intp)
                    g2 = X_c[:, a1_arr].T @ X_c[:, j2] if gram_free else G_c[a1_arr, j2]
                    v2 = A1 @ g2
                    s2 = G_c_diag_reg[j2] - g2 @ v2
                    if s2 < 1e-12:
                        continue
                    d2 = Xy_c[j2] - g2 @ u1
                    cl2 = d2 / s2
                    # Compute 2-step R² (train only, fast path)
                    if not use_val:
                        u1_rhs = float(u1 @ Xy_c[a1_arr])
                        u1_nsq = float(u1 @ u1)
                        u1_dot_v2 = float(u1 @ v2)
                        v2_nsq = float(v2 @ v2)
                        coef2_nsq = u1_nsq - 2.0 * cl2 * u1_dot_v2 + cl2 * cl2 * (v2_nsq + 1.0)
                        ss2 = yy_c - (u1_rhs + d2 * cl2) - alpha * coef2_nsq
                        pair_r2 = 1.0 - ss2 / ss_tot_train
                    else:
                        # Val path: full prediction
                        a2_arr = np.array(active_idx + [j1, j2], dtype=np.intp)
                        k2 = k1 + 1
                        A2 = np.empty((k2, k2))
                        s2_inv = 1.0 / s2
                        A2[:k1, :k1] = A1 + s2_inv * np.outer(v2, v2)
                        A2[:k1, k1] = -s2_inv * v2
                        A2[k1, :k1] = -s2_inv * v2
                        A2[k1, k1] = s2_inv
                        u2 = A2 @ Xy_c[a2_arr]
                        intercept2 = y_mean - mu[a2_arr] @ u2
                        pred2 = X_val[:, a2_arr] @ u2 + intercept2
                        ss2 = ((y_val - pred2) ** 2).sum()
                        pair_r2 = 1.0 - ss2 / ss_tot_val

                    if pair_r2 > best_pair_r2:
                        best_pair_r2 = pair_r2
                        best_local = t1
                        best_j = j1

            best_r2_raw = float(r2_all[best_local])

        best_r2 = best_r2_raw

        if not valid[best_local]:
            break

        marginal = best_r2 - prev_r2
        if min_delta_r2 > 0 and marginal < min_delta_r2:
            if verbose:
                print(f'    step {step+1}: marginal R²={marginal:.6f} '
                      f'< min_delta_r2={min_delta_r2:.6f}, stopping')
            break

        # Update A_inv via block matrix inversion (rank-1 extension)
        if k == 0:
            A_inv = np.array([[1.0 / G_c_diag_reg[best_j]]])
        else:
            g = X_c[:, active_arr].T @ X_c[:, best_j] if gram_free else G_c[active_arr, best_j]
            v = A_inv @ g
            s_inv = 1.0 / (G_c_diag_reg[best_j] - g @ v)
            k1 = k + 1
            A_inv_new = np.empty((k1, k1))
            A_inv_new[:k, :k] = A_inv + s_inv * np.outer(v, v)
            A_inv_new[:k, k] = -s_inv * v
            A_inv_new[k, :k] = -s_inv * v
            A_inv_new[k, k] = s_inv
            A_inv = A_inv_new

        active_idx.append(best_j)
        remaining[best_j] = False

        # Update cached quantities
        active_arr_new = np.array(active_idx, dtype=np.intp)
        u = A_inv @ Xy_c[active_arr_new]
        u_rhs = float(u @ Xy_c[active_arr_new])
        u_norm_sq = float(u @ u)

        history.append(
            library.names[best_j],
            library.families[best_j],
            marginal,
            best_r2,
        )
        prev_r2 = best_r2

        if verbose:
            elapsed = time.time() - step_t0
            total = time.time() - t0
            val_tag = ' (val)' if use_val else ''
            print(f'    step {step+1:>2}/{max_steps}: +{library.names[best_j]:<30} '
                  f'R²={best_r2:.4f} (+{marginal:.4f}){val_tag}  '
                  f'[{elapsed:.1f}s / {total:.1f}s total]',
                  flush=True)

        # Parent retention: inject component features of the selected compound
        if parent_retain > 0 and marginal > 1e-8:
            min_parent_gain = parent_retain * marginal
            components = _find_component_names(
                library.names[best_j], library.names)

            for comp_name in components:
                # Find index of this component
                comp_idx = None
                for ci, cn in enumerate(library.names):
                    if cn == comp_name and remaining[ci]:
                        comp_idx = ci
                        break
                if comp_idx is None:
                    continue

                # Compute marginal R² for this parent (train-based)
                kp = len(active_idx)
                active_arr_p = np.array(active_idx, dtype=np.intp)
                gp = X_c[:, active_arr_p].T @ X_c[:, comp_idx] if gram_free else G_c[active_arr_p, comp_idx]
                vp = A_inv @ gp
                sp = G_c_diag_reg[comp_idx] - gp @ vp
                if sp < 1e-12:
                    continue
                dp = Xy_c[comp_idx] - gp @ u
                clp = dp / sp

                u_dot_vp = u @ vp
                vp_nsq = vp @ vp
                coef_nsq_p = (u_norm_sq - 2.0 * clp * u_dot_vp
                              + clp * clp * (vp_nsq + 1.0))
                ss_res_p = yy_c - (u_rhs + dp * clp) - alpha * coef_nsq_p
                parent_r2 = 1.0 - ss_res_p / ss_tot_train
                parent_marginal = parent_r2 - prev_r2

                if parent_marginal < min_parent_gain:
                    continue

                # Inject this parent into the active set
                sp_inv = 1.0 / sp
                kp1 = kp + 1
                A_inv_new = np.empty((kp1, kp1))
                A_inv_new[:kp, :kp] = A_inv + sp_inv * np.outer(vp, vp)
                A_inv_new[:kp, kp] = -sp_inv * vp
                A_inv_new[kp, :kp] = -sp_inv * vp
                A_inv_new[kp, kp] = sp_inv
                A_inv = A_inv_new

                active_idx.append(comp_idx)
                remaining[comp_idx] = False

                active_arr_new = np.array(active_idx, dtype=np.intp)
                u = A_inv @ Xy_c[active_arr_new]
                u_rhs = float(u @ Xy_c[active_arr_new])
                u_norm_sq = float(u @ u)

                history.append(
                    library.names[comp_idx],
                    library.families[comp_idx],
                    parent_marginal,
                    parent_r2,
                )
                prev_r2 = parent_r2

                if verbose:
                    print(f'      +parent: +{library.names[comp_idx]:<28} '
                          f'R²={parent_r2:.4f} (+{parent_marginal:.4f})',
                          flush=True)

        if best_r2 > r2_stop or prev_r2 > r2_stop:
            break

    return history


def correlation_select(library, y, max_features=5, y_val=None,
                       diversity=0.5, verbose=False):
    """mRMR-style selection by univariate |correlation| with target.

    score(j) = |corr(j, y)| - diversity * mean(|corr(j, s)| for s in selected)

    Parameters
    ----------
    library : PrimitiveLibrary
    y : array-like of shape (n_samples,)
    max_features : int
    y_val : array-like, optional
        Use val correlations for ranking.
    diversity : float
        Redundancy penalty weight (0=off, 1=full mRMR).
    verbose : bool

    Returns
    -------
    SelectionHistory (cumulative_r2 values are mRMR scores, not model R^2)
    """
    y = np.asarray(y, dtype=np.float64)

    use_val = library.X_val is not None and y_val is not None
    if use_val:
        X_score = library.X_val
        y_score = np.asarray(y_val, dtype=np.float64)
    else:
        X_score = library.X
        y_score = y

    n_features = X_score.shape[1]

    # Precompute relevance: |corr(feature, y)| — vectorized
    x_means = X_score.mean(axis=0)
    x_stds = X_score.std(axis=0)
    y_std = y_score.std()
    valid_x = x_stds > 1e-10
    x_stds_safe = x_stds.copy()
    x_stds_safe[~valid_x] = 1.0
    n_obs = X_score.shape[0]
    X_c = X_score - x_means
    y_c = y_score - y_score.mean()
    corrs = np.abs(X_c.T @ y_c) / (x_stds_safe * y_std * n_obs)
    corrs[~valid_x] = 0.0

    k = min(max_features, n_features)
    selected = []
    remaining = set(range(n_features))
    history = SelectionHistory('mrmr', 'validation' if library.X_val is not None and y_val is not None else 'train')

    # Precompute normalized features for fast inter-correlation
    if diversity > 0:
        X_norm = (X_score - x_means) / x_stds_safe
        # Cache full correlation matrix for O(1) lookup
        gram = X_norm.T @ X_norm / n_obs

    for rank in range(k):
        best_j = -1
        best_score = -np.inf

        remaining_arr = np.array(list(remaining))
        remaining_corrs = corrs[remaining_arr]
        nonzero_mask = remaining_corrs > 0
        if not nonzero_mask.any():
            break

        if diversity > 0 and selected:
            sel_arr = np.array(selected)
            redundancy = np.abs(gram[np.ix_(remaining_arr, sel_arr)]).mean(axis=1)
            scores = remaining_corrs - diversity * redundancy
        else:
            scores = remaining_corrs.copy()

        scores[~nonzero_mask] = -np.inf
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_j = remaining_arr[best_idx]
        if best_score == -np.inf:
            best_j = -1

        if best_j == -1:
            break

        selected.append(best_j)
        remaining.discard(best_j)

        history.append(
            library.names[best_j],
            library.families[best_j],
            corrs[best_j],
            best_score,
        )
        if verbose:
            val_tag = ' (val)' if use_val else ''
            red_str = f'  score={best_score:.4f}' if diversity > 0 else ''
            print(f'    {rank+1:>2}/{k}: |r|={corrs[best_j]:.4f}{red_str}  '
                  f'{library.names[best_j]}{val_tag}')

    return history


def residual_select(library, y, max_features=20, y_val=None, verbose=False,
                    family_diversity=0.0, lookahead=False,
                    group_ids=None, group_weight=0.0,
                    _gram_free='auto'):
    """OMP-style selection by correlation with residual.

    At each step: fit y ~ selected features (OLS), compute residual,
    pick the candidate most correlated with the residual.

    Parameters
    ----------
    library : PrimitiveLibrary
    y : array-like of shape (n_samples,)
    max_features : int
    y_val : array-like, optional
        Report R² on library.X_val and these targets. Feature ranking still
        uses training residuals; validation labels do not choose candidates.
    verbose : bool
    family_diversity : float
        Soft penalty for same-family redundancy (0=off).
    lookahead : bool
        If True, check top-8 pairs for synergistic 2-step picks.
    group_ids : array-like of shape (n_samples,), optional
        Group labels (e.g., dataset/modality IDs). When provided with
        group_weight > 0, scoring blends global correlation with
        mean per-group correlation, favoring cross-group features.
    group_weight : float
        Blend weight for group-aware scoring (0=global only, 1=group-mean
        only). Default 0.0 preserves original behavior.

    Returns
    -------
    SelectionHistory
    """
    from sklearn.metrics import r2_score

    y = np.asarray(y, dtype=np.float64)
    X = library.X
    n_samples, n_features = X.shape

    k = min(max_features, n_features)
    selected_idx = []
    history = SelectionHistory('r2', 'validation' if library.X_val is not None and y_val is not None else 'train')
    residual = y - y.mean()

    stds = X.std(axis=0)
    valid = stds > 1e-10
    X_means = X.mean(axis=0)
    X_centered = X - X_means
    y_mean = y.mean()
    y_centered = y - y_mean
    stds_safe = np.where(valid, stds, 1.0)


    # Resolve gram-free mode: skip d×d Gram for large d
    if _gram_free == 'auto':
        gram_free = n_features >= _GRAM_FREE_THRESHOLD
    else:
        gram_free = bool(_gram_free)

    if not gram_free:
        # Precompute full Gram and X'y for OLS refit via normal equations
        G = X_centered.T @ X_centered                  # (d, d)
        Xy = X_centered.T @ y_centered                 # (d,)

    # Group-aware scoring setup
    use_groups = group_ids is not None and group_weight > 0
    if use_groups:
        group_ids = np.asarray(group_ids)
        unique_groups = np.unique(group_ids)
        group_masks = [(group_ids == g) for g in unique_groups]
        group_ns = [int(gm.sum()) for gm in group_masks]
        group_X_c = [X_centered[gm] for gm in group_masks]
        group_stds = [X[gm].std(axis=0) for gm in group_masks]
        for gs in group_stds:
            gs[gs < 1e-10] = 1.0

    for rank in range(k):
        r_std = residual.std()
        if r_std < 1e-10:
            break

        # Vectorized correlation: |corr(X[:,j], residual)| for all j
        r_centered = residual - residual.mean()
        corrs = np.abs(X_centered.T @ r_centered) / (stds_safe * r_std * n_samples)

        # Group-aware blending
        if use_groups:
            group_corrs = np.zeros(n_features)
            n_valid_groups = 0
            for gm, gXc, gs, gn in zip(group_masks, group_X_c, group_stds, group_ns):
                r_g = residual[gm]
                r_g_std = r_g.std()
                if r_g_std < 1e-10 or gn < 10:
                    continue
                r_g_c = r_g - r_g.mean()
                gc_val = np.abs(gXc.T @ r_g_c) / (gs * r_g_std * gn)
                group_corrs += gc_val
                n_valid_groups += 1
            if n_valid_groups > 0:
                group_corrs /= n_valid_groups
                corrs = (1 - group_weight) * corrs + group_weight * group_corrs

        # Mask out selected and invalid features
        mask = np.ones(n_features, dtype=bool)
        if selected_idx:
            mask[selected_idx] = False
        mask[~valid] = False
        corrs[~mask] = -1.0

        # Family diversity: penalize over-represented families
        scores = corrs.copy()
        if family_diversity > 0 and selected_idx:
            fam_count = {}
            for si in selected_idx:
                fam = library.families[si]
                fam_count[fam] = fam_count.get(fam, 0) + 1
            n_sel = len(selected_idx)
            for j in range(n_features):
                if not mask[j]:
                    continue
                fam = library.families[j]
                cnt = fam_count.get(fam, 0)
                penalty = family_diversity * (cnt / n_sel) * scores[j] * 0.3
                scores[j] -= penalty

        best_j = int(np.argmax(scores))
        best_corr = corrs[best_j]
        if best_corr < 0:
            break

        # 2-step lookahead: test if a different pick leads to better 2-step R²
        if lookahead and rank < k - 1:
            top_n = min(8, int(mask.sum()))
            if top_n > 1:
                top_cands = np.argsort(-scores)[:top_n]
                top_cands = [j for j in top_cands if mask[j] and corrs[j] > 0]
                best_2step_var = np.inf
                best_2step_j = best_j

                for j1 in top_cands:
                    # Simulate picking j1: refit OLS with selected + j1
                    trial = selected_idx + [j1]
                    sa = np.array(trial)
                    if gram_free:
                        coef_t = np.linalg.lstsq(X_centered[:, sa], y_centered, rcond=None)[0]
                    else:
                        G_t = G[np.ix_(sa, sa)] + 1e-6 * np.eye(len(trial))
                        coef_t = np.linalg.lstsq(G_t, Xy[sa], rcond=None)[0]
                    resid1 = y_centered - X_centered[:, sa] @ coef_t

                    # Find best 2nd pick from residual of j1
                    r1_std = resid1.std()
                    if r1_std < 1e-10:
                        var_2step = 0.0
                    else:
                        r1_c = resid1 - resid1.mean()
                        corrs2 = np.abs(X_centered.T @ r1_c) / (stds_safe * r1_std * n_samples)
                        mask2 = mask.copy()
                        mask2[j1] = False
                        corrs2[~mask2] = -1.0
                        j2 = int(np.argmax(corrs2))
                        if corrs2[j2] > 0:
                            trial2 = trial + [j2]
                            sa2 = np.array(trial2)
                            if gram_free:
                                coef_t2 = np.linalg.lstsq(X_centered[:, sa2], y_centered, rcond=None)[0]
                            else:
                                G_t2 = G[np.ix_(sa2, sa2)] + 1e-6 * np.eye(len(trial2))
                                coef_t2 = np.linalg.lstsq(G_t2, Xy[sa2], rcond=None)[0]
                            resid2 = y_centered - X_centered[:, sa2] @ coef_t2
                            var_2step = np.var(resid2)
                        else:
                            var_2step = np.var(resid1)

                    if var_2step < best_2step_var:
                        best_2step_var = var_2step
                        best_2step_j = j1

                best_j = best_2step_j
                best_corr = corrs[best_j]

        selected_idx.append(best_j)

        # Refit OLS
        sel_arr = np.array(selected_idx)
        n_sel = len(selected_idx)

        if gram_free:
            coef = np.linalg.lstsq(X_centered[:, sel_arr], y_centered, rcond=None)[0]
        else:
            G_sel = G[np.ix_(sel_arr, sel_arr)] + 1e-6 * np.eye(n_sel)
            Xy_sel = Xy[sel_arr]
            coef = np.linalg.lstsq(G_sel, Xy_sel, rcond=None)[0]
        y_hat = X_centered[:, sel_arr] @ coef + y_mean
        residual = y - y_hat

        if library.X_val is not None and y_val is not None:
            val_hat = ((library.X_val[:, sel_arr] - X_means[sel_arr])
                       @ coef + y_mean)
            r2 = r2_score(y_val, val_hat)
        else:
            r2 = r2_score(y, y_hat)

        history.append(
            library.names[best_j],
            library.families[best_j],
            r2 - (history.steps[-1]['cumulative_r2'] if history.steps else 0.0),
            r2,
            selection_score=best_corr,
        )
        if verbose:
            print(f'    {rank+1:>2}/{k}: |r_corr|={best_corr:.4f}  R²={r2:.4f}  '
                  f'{library.names[best_j]}')

    return history


def per_class_select(library, y, method='omp', max_features=20,
                     y_val=None, verbose=False, **kwargs):
    """One-vs-all feature selection for classification targets.

    Runs the specified selection method independently for each class
    (binary target: class vs rest), returning per-class histories.

    Parameters
    ----------
    library : PrimitiveLibrary
    y : array-like of shape (n_samples,)
        Integer class labels.
    method : str
        'omp', 'greedy', or 'correlation'.
    max_features : int
    y_val : array-like, optional
    verbose : bool
    **kwargs : passed to the underlying selection function.

    Returns
    -------
    dict mapping class_label -> SelectionHistory
    """
    y = np.asarray(y)
    classes = sorted(set(int(c) for c in y))

    select_fn = {
        'omp': residual_select,
        'greedy': greedy_forward_select,
        'correlation': correlation_select,
    }
    if method not in select_fn:
        raise ValueError(f"Unknown method '{method}', expected one of {list(select_fn)}")
    fn = select_fn[method]

    results = {}
    for cls in classes:
        y_bin = (y == cls).astype(np.float64)
        y_val_bin = None
        if y_val is not None:
            y_val_arr = np.asarray(y_val)
            y_val_bin = (y_val_arr == cls).astype(np.float64)

        if verbose:
            print(f'  Class {cls} ({int(y_bin.sum())}/{len(y_bin)} positive):')

        results[cls] = fn(library, y_bin, max_features=max_features,
                          y_val=y_val_bin, verbose=verbose, **kwargs)

    return results

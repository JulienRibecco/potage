"""Replay a fitted stage graph using recorded training references."""
import numpy as np
from ._records import FeatureSet
from ._algebra import build_stage0_raw, build_stage1_am
from ._carriers import build_stage2_static_carriers, _build_targeted_carriers
from ._feature_utils import _derive_masks
from ._modulation import (
    _fm_sin_kernel,
    _pm_sin_kernel,
    _fm_tri_kernel,
    _fm_pulse_kernel,
    _wm_center_kernel,
    _wm_width_kernel,
    _apply_modulation,
)
from ._fields import build_stage_fields, build_probe_zones
from ._kfold import _subset_by_names
from ._cross import _apply_cross_modulation, _build_cross_am


class ReplayMixin:
    # -- transform -----------------------------------------------------------

    def transform(self, X_new):
        """Replay the recorded pipeline on new data.

        Regenerates candidates using the same builder functions and
        subsets to the features selected during training. Returns the
        output corresponding to the final stage.

        Parameters
        ----------
        X_new : ndarray of shape (n_new, n_features)
            New data with the same columns as the original ``X``.

        Returns
        -------
        ndarray of shape (n_new, n_final_features)
        """
        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"X_new has {X_new.shape[1]} columns, expected {self.X.shape[1]}"
            )
        if not self._stages:
            raise ValueError("No stages recorded — nothing to transform")

        n_new = X_new.shape[0]
        cache = {}  # stage_index → transformed X_new for that stage

        for si, stage in enumerate(self._stages):
            stype = stage._step_type

            if stype == 'raw':
                X_out = self._transform_raw(X_new, stage, n_new)
            elif stype == 'am_single':
                parent_X = cache[stage._parent_ids[0]]
                parent_stage = self._stages[stage._parent_ids[0]]
                X_out = self._transform_am_single(parent_X, parent_stage, stage, n_new)
            elif stype == 'am_cross':
                parent_Xs = [cache[pi] for pi in stage._parent_ids]
                parent_stages = [self._stages[pi] for pi in stage._parent_ids]
                X_out = self._transform_am_cross(parent_Xs, parent_stages, stage, n_new)
            elif stype in ('carriers', 'fft_carriers'):
                parent_X = cache[stage._parent_ids[0]]
                parent_stage = self._stages[stage._parent_ids[0]]
                X_out = self._transform_carriers(parent_X, parent_stage, stage, n_new)
            elif stype == 'fm':
                carrier_X = cache[stage._parent_ids[0]]
                mod_X = cache[stage._parent_ids[1]]
                carrier_stage = self._stages[stage._parent_ids[0]]
                mod_stage = self._stages[stage._parent_ids[1]]
                X_out = self._transform_fm(
                    carrier_X, carrier_stage, mod_X, mod_stage, stage, n_new)
            elif stype == 'wm':
                carrier_X = cache[stage._parent_ids[0]]
                mod_X = cache[stage._parent_ids[1]]
                carrier_stage = self._stages[stage._parent_ids[0]]
                mod_stage = self._stages[stage._parent_ids[1]]
                X_out = self._transform_wm(
                    carrier_X, carrier_stage, mod_X, mod_stage, stage, n_new)
            elif stype == 'fm_nonsmooth':
                carrier_X = cache[stage._parent_ids[0]]
                mod_X = cache[stage._parent_ids[1]]
                carrier_stage = self._stages[stage._parent_ids[0]]
                mod_stage = self._stages[stage._parent_ids[1]]
                X_out = self._transform_fm_nonsmooth(
                    carrier_X, carrier_stage, mod_X, mod_stage, stage, n_new)
            elif stype == 'select':
                parent_X = cache[stage._parent_ids[0]]
                X_out = self._transform_select(parent_X, stage, n_new)
            elif stype == 'deepen_gen':
                parent_X = cache[stage._parent_ids[0]]
                parent_stage = self._stages[stage._parent_ids[0]]
                X_out = self._transform_deepen_gen(
                    parent_X, parent_stage, stage, n_new)
            elif stype == 'fields':
                parent_X = cache[stage._parent_ids[0]]
                parent_stage = self._stages[stage._parent_ids[0]]
                X_out = self._transform_fields(parent_X, parent_stage, stage, n_new)
            elif stype == 'probe_fields':
                parent_X = cache[stage._parent_ids[0]]
                parent_stage = self._stages[stage._parent_ids[0]]
                X_out = self._transform_probe_fields(parent_X, parent_stage, stage, n_new)
            elif stype == 'fuse':
                parent_Xs = [cache[pi] for pi in stage._parent_ids]
                X_out = np.hstack(parent_Xs)
            else:
                raise ValueError(f"Unknown step type: {stype!r}")

            cache[si] = X_out

        # Return the last stage's output
        return cache[len(self._stages) - 1]

    def _get_target_names(self, stage):
        """Get the feature names to subset to for transform replay."""
        if stage.history is not None:
            return stage.history.selected_names()
        if stage._surviving_names is not None:
            return stage._surviving_names
        return stage.output.names

    def _transform_raw(self, X_new, stage, n_new):
        """Replay raw() on new data."""
        is_numeric, _, _ = _derive_masks(self.feature_types)
        cols, pnames, _ = build_stage0_raw(
            X_new, self.names, is_numeric,
            reference_X=self.X,
            feature_types=self.feature_types,
        )
        selected = self._get_target_names(stage)
        return _subset_by_names(cols, pnames, selected, n_new)

    def _transform_am_single(self, parent_X_new, parent_stage, stage, n_new):
        """Replay single-set AM on new data."""
        parent_names = parent_stage.output.names
        parent_types = parent_stage.output.feature_types
        is_numeric, _, _ = _derive_masks(parent_types)
        cols, pnames, _ = build_stage1_am(
            parent_X_new, parent_names, is_numeric,
            config=self.config,
            feature_types=parent_types, reference_X=parent_stage.output.X,
        )
        selected = self._get_target_names(stage)
        return _subset_by_names(cols, pnames, selected, n_new)

    def _transform_am_cross(self, parent_Xs, parent_stages, stage, n_new):
        """Replay cross-set AM on new data."""
        sets = []
        for px, ps in zip(parent_Xs, parent_stages):
            sets.append(FeatureSet(px, ps.output.names, ps.output.feature_types))
        # Compute train stds for deterministic ratio filtering
        train_X = np.hstack([self._stages[pi].output.X
                             for pi in stage._parent_ids])
        train_stds = train_X.std(axis=0)
        cols, pnames, _ = _build_cross_am(sets, self.config, ref_stds=train_stds,
                                          reference_X=train_X)
        selected = self._get_target_names(stage)
        return _subset_by_names(cols, pnames, selected, n_new)

    def _transform_carriers(self, parent_X_new, parent_stage, stage, n_new):
        """Replay carriers() or fft_carriers() on new data."""
        parent_names = parent_stage.output.names
        parent_types = parent_stage.output.feature_types
        is_numeric, _, _ = _derive_masks(parent_types)
        # Use parent's train data as reference for norm01 stats
        reference_X = parent_stage.output.X

        # Check if this was an fft_carriers stage with stored periodicities
        meta = stage._metadata or {}
        periodicities = meta.get('fft_periodicities')
        if periodicities is not None:
            cols, pnames, _ = _build_targeted_carriers(
                parent_X_new, parent_names, is_numeric, self.config,
                periodicities, reference_X=reference_X,
            )
        else:
            cols, pnames, _ = build_stage2_static_carriers(
                parent_X_new, parent_names, is_numeric, self.config,
                reference_X=reference_X,
                feature_space=meta.get('feature_space', 'linear'),
            )
        selected = self._get_target_names(stage)
        return _subset_by_names(cols, pnames, selected, n_new)

    def _transform_fields(self, parent_X_new, parent_stage, stage, n_new):
        """Replay fields() on new data."""
        parent_names = parent_stage.output.names
        parent_types = parent_stage.output.feature_types
        is_numeric, _, _ = _derive_masks(parent_types)
        reference_X = parent_stage.output.X

        cols, pnames, _ = build_stage_fields(
            parent_X_new, parent_names, is_numeric, self.config,
            reference_X=reference_X,
        )
        selected = self._get_target_names(stage)
        return _subset_by_names(cols, pnames, selected, n_new)

    def _transform_probe_fields(self, parent_X_new, parent_stage, stage, n_new):
        """Replay probe_fields() on new data using stored groups."""
        parent_names = parent_stage.output.names
        reference_X = parent_stage.output.X

        meta = stage._metadata or {}
        groups = meta.get('probe_groups', [])

        cols, pnames, _ = build_probe_zones(
            parent_X_new, parent_names, groups, config=self.config,
            reference_X=reference_X,
        )
        selected = self._get_target_names(stage)
        return _subset_by_names(cols, pnames, selected, n_new)

    def _transform_fm(self, carrier_X_new, carrier_stage,
                      mod_X_new, mod_stage, stage, n_new):
        """Replay FM/PM modulation on new data."""
        fm_grid = [{'k': k, 'd': d}
                   for k in self.config.carrier_freqs
                   for d in self.config.fm_depths]
        pm_grid = [{'k': k, 'p': p}
                   for k in self.config.carrier_freqs
                   for p in self.config.pm_depths]
        kernels = [
            (_fm_sin_kernel, fm_grid, 'fm_sin'),
            (_pm_sin_kernel, pm_grid, 'pm_sin'),
        ]
        # Use train data as reference for norm01 stats
        cols, pnames, _ = _apply_cross_modulation(
            carrier_X_new, carrier_stage.output.names,
            carrier_stage.output.feature_types,
            mod_X_new, mod_stage.output.names,
            mod_stage.output.feature_types,
            self.config, kernels,
            ref_X_c=carrier_stage.output.X,
            ref_X_m=mod_stage.output.X,
        )
        selected = self._get_target_names(stage)
        return _subset_by_names(cols, pnames, selected, n_new)

    def _transform_wm(self, carrier_X_new, carrier_stage,
                      mod_X_new, mod_stage, stage, n_new):
        """Replay WM modulation on new data."""
        center_grid = [{'q': q} for q in self.config.wm_center_pcts]
        width_grid = [{'q': q, 'wmod': wmod}
                      for q in self.config.wm_center_pcts
                      for wmod in self.config.wm_width_mods]
        kernels = [
            (_wm_center_kernel, center_grid, 'wm_center'),
            (_wm_width_kernel, width_grid, 'wm_width'),
        ]
        cols, pnames, _ = _apply_cross_modulation(
            carrier_X_new, carrier_stage.output.names,
            carrier_stage.output.feature_types,
            mod_X_new, mod_stage.output.names,
            mod_stage.output.feature_types,
            self.config, kernels,
            ref_X_c=carrier_stage.output.X,
            ref_X_m=mod_stage.output.X,
        )
        selected = self._get_target_names(stage)
        return _subset_by_names(cols, pnames, selected, n_new)

    def _transform_fm_nonsmooth(self, carrier_X_new, carrier_stage,
                                mod_X_new, mod_stage, stage, n_new):
        """Replay FM nonsmooth modulation on new data."""
        fm_tri_grid = [{'k': k, 'd': d}
                       for k in self.config.carrier_freqs
                       for d in self.config.fm_depths]
        fm_pulse_grid = [{'q': q, 'w': w}
                         for q in self.config.gauss_center_pcts
                         for w in self.config.gauss_widths]
        kernels = [
            (_fm_tri_kernel, fm_tri_grid, 'fm_tri'),
            (_fm_pulse_kernel, fm_pulse_grid, 'fm_pulse'),
        ]
        cols, pnames, _ = _apply_cross_modulation(
            carrier_X_new, carrier_stage.output.names,
            carrier_stage.output.feature_types,
            mod_X_new, mod_stage.output.names,
            mod_stage.output.feature_types,
            self.config, kernels,
            ref_X_c=carrier_stage.output.X,
            ref_X_m=mod_stage.output.X,
        )
        selected = self._get_target_names(stage)
        return _subset_by_names(cols, pnames, selected, n_new)

    def _transform_select(self, parent_X_new, stage, n_new):
        """Replay select() — subset parent by selected names."""
        selected = self._get_target_names(stage)
        parent_stage = self._stages[stage._parent_ids[0]]
        parent_names = parent_stage.output.names
        name_to_idx = {n: i for i, n in enumerate(parent_names)}
        idx = [name_to_idx[n] for n in selected if n in name_to_idx]
        return parent_X_new[:, idx]

    def _transform_deepen_gen(self, carry_X_new, carry_stage, stage, n_new):
        """Replay deepen modulation generation on new data."""
        meta = stage._metadata
        cfg = self.config
        carry_names = carry_stage.output.names
        is_numeric = np.ones(len(carry_names), dtype=bool)
        top_idx = np.arange(len(carry_names))

        fm_grid = [{'k': k, 'd': d}
                   for k in meta['fm_ks'] for d in meta['fm_ds']]
        pm_grid = [{'k': k, 'p': p}
                   for k in meta['pm_ks'] for p in meta['pm_ps']]
        wm_center_grid = [{'q': q} for q in meta['wm_qs']]
        wm_width_grid = [{'q': q, 'wmod': wmod}
                         for q in meta['wm_qs']
                         for wmod in cfg.wm_width_mods]
        depth = meta['depth']
        kernels = [
            (_fm_sin_kernel, fm_grid, f'L{depth}_fm'),
            (_pm_sin_kernel, pm_grid, f'L{depth}_pm'),
            (_wm_center_kernel, wm_center_grid, f'L{depth}_wm_center'),
            (_wm_width_kernel, wm_width_grid, f'L{depth}_wm_width'),
        ]

        cols, pnames, _ = _apply_modulation(
            carry_X_new, carry_names, is_numeric, cfg, top_idx, kernels,
            reference_X=carry_stage.output.X)

        # Subset to surviving names
        selected = self._get_target_names(stage)
        return _subset_by_names(cols, pnames, selected, n_new)

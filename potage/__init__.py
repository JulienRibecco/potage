"""potage -- Domain-agnostic feature engineering via carrier x modulator stages.

Build diverse feature libraries from raw tabular data using progressive
signal-inspired transformations (products, sinusoidal carriers, FM/PM modulation,
Gaussian windows), then select the most predictive subset via greedy/mRMR/OMP.

Core classes:
    Pipeline           -- Modular pipeline builder with composable feature set handles
    Config             -- Configuration for frequency grids and modulation depths
    PrimitiveLibrary   -- Feature matrix container with names + families
    SelectionHistory   -- Ordered steps with explicit score kind and split
    FeatureSet         -- Immutable handle to a pipeline step's output

Selection algorithms:
    greedy_forward_select  -- Ridge-based forward selection (default)
    correlation_select     -- mRMR-style univariate selection
    residual_select        -- OMP-style residual correlation
    per_class_select       -- One-vs-all wrapper for classification

Diagnosis:
    diagnose_curve     -- Categorize R^2 curve shape

Optional native builders (import from potage._accel; external libsoup required):
    HAS_C_SOUP, build_lib_c, build_lib_deep_c, c_omp_select
"""

from ._library import PrimitiveLibrary, SelectionHistory, warn_rank_features
from ._select import (
    greedy_forward_select,
    correlation_select,
    residual_select,
    per_class_select,
)
from ._diagnose import diagnose_curve
from ._config import Config, SoupConfig
from ._stages import apply_stage
from ._frequency import fft_freqs
from ._fields import (
    build_stage_fields,
    build_zone_adjustments,
    build_field_pipeline,
    neuron_probe_groups,
    build_probe_zones,
)
from ._dedup import StructuralDedup
from ._pipe import (
    Pipeline,
    SoupPipe,
    FeatureSet,
    CandidateBudgetExceeded,
    MemoryBudgetExceeded,
    Stage,
)
from ._presets import (
    multi_pass,
    MultiPassResult,
    FeaturePool,
    StageResult,
    recipe_light,
    recipe_std,
    recipe_am,
    recipe_deep,
    recipe_carrier_first,
    recipe_explore,
    recipe_explore_deep,
    recipe_tower,
    recipe_tower_wm,
)

__all__ = [
    # Core
    'Pipeline',
    'Config',
    'PrimitiveLibrary',
    'SelectionHistory',
    'FeatureSet',
    'Stage',
    # Selection
    'greedy_forward_select',
    'correlation_select',
    'residual_select',
    'per_class_select',
    # Diagnosis
    'diagnose_curve',
    # Stage builders
    'apply_stage',
    'fft_freqs',
    'build_stage_fields',
    'build_field_pipeline',
    'neuron_probe_groups',
    'build_probe_zones',
    # Presets
    'multi_pass',
    'MultiPassResult',
    'FeaturePool',
    'StageResult',
    'recipe_light',
    'recipe_std',
    'recipe_am',
    'recipe_deep',
    'recipe_carrier_first',
    'recipe_explore',
    'recipe_explore_deep',
    'recipe_tower',
    'recipe_tower_wm',
    # Utilities
    'StructuralDedup',
    'CandidateBudgetExceeded',
    'MemoryBudgetExceeded',
    'warn_rank_features',
    # Backward compat aliases
    'SoupPipe',
    'SoupConfig',
]

"""Optional native feature builders and a Python OMP compatibility wrapper.

Searches for libsoup in:
  1. SIGNALFAULT_SOUP_LIB environment variable
  2. This module's directory (potage/)
  3. A legacy sibling soup/ directory

Native builders require libsoup. c_omp_select always delegates to Python.
Load an externally built libsoup via SIGNALFAULT_SOUP_LIB.
Native builders use the native defaults; Python Config is not supported.

Usage:
    from potage._accel import HAS_C_SOUP, build_lib_c, c_omp_select
"""

import os
import ctypes
import numpy as np
from ctypes import c_int, c_double, POINTER, c_char, c_void_p

from ._library import PrimitiveLibrary
from ._select import residual_select

SOUP_MAX_NAME_LEN = 128
SOUP_MAX_FAMILY_LEN = 32

# ======================================================================
# Load shared library
# ======================================================================

_lib = None


def _find_libsoup():
    """Search for libsoup shared library in standard locations."""
    candidates = []

    # 1. Environment variable
    env_path = os.environ.get('SIGNALFAULT_SOUP_LIB')
    if env_path:
        candidates.append(env_path)

    # 2. This module's directory
    module_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ('libsoup.dylib', 'libsoup.so'):
        candidates.append(os.path.join(module_dir, name))

    # 3. Legacy sibling soup/ directory
    soup_dir = os.path.join(os.path.dirname(module_dir), 'soup')
    for name in ('libsoup.dylib', 'libsoup.so'):
        candidates.append(os.path.join(soup_dir, name))

    for path in candidates:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    return None


_lib = _find_libsoup()
HAS_C_SOUP = _lib is not None

if _lib:
    # Build and clean (train)
    _lib.soup_lib_build_and_clean.restype = c_void_p
    _lib.soup_lib_build_and_clean.argtypes = [
        POINTER(c_double), c_int, c_int,
        c_void_p,
        POINTER(c_int),
        c_void_p,
        c_double, c_double,
        c_void_p,
    ]

    # Build val
    _lib.soup_lib_build_val.restype = c_void_p
    _lib.soup_lib_build_val.argtypes = [
        POINTER(c_double), c_int, c_int,
        c_void_p, POINTER(c_int), c_void_p,
        c_void_p,
        c_void_p, c_int,
    ]

    # OMP
    _lib.soup_lib_omp_select.restype = c_void_p
    _lib.soup_lib_omp_select.argtypes = [c_void_p, POINTER(c_double), c_int]

    # Accessors
    _lib.soup_lib_ncols.restype = c_int
    _lib.soup_lib_ncols.argtypes = [c_void_p]
    _lib.soup_lib_nsamples.restype = c_int
    _lib.soup_lib_nsamples.argtypes = [c_void_p]
    _lib.soup_lib_data.restype = POINTER(c_double)
    _lib.soup_lib_data.argtypes = [c_void_p]
    _lib.soup_lib_names.restype = c_void_p
    _lib.soup_lib_names.argtypes = [c_void_p]
    _lib.soup_lib_families.restype = c_void_p
    _lib.soup_lib_families.argtypes = [c_void_p]
    _lib.soup_lib_free_matrix.restype = None
    _lib.soup_lib_free_matrix.argtypes = [c_void_p]
    _lib.soup_lib_free_stats.restype = None
    _lib.soup_lib_free_stats.argtypes = [c_void_p]

    # Build deep (stages 0-3/4)
    _lib.soup_lib_build_deep.restype = c_void_p
    _lib.soup_lib_build_deep.argtypes = [
        POINTER(c_double), c_int, c_int,
        c_void_p, POINTER(c_int), c_void_p,
        POINTER(c_int), c_int,
        c_int,
        c_double, c_double,
        c_void_p,
    ]

    _lib.soup_lib_build_deep_val.restype = c_void_p
    _lib.soup_lib_build_deep_val.argtypes = [
        POINTER(c_double), c_int, c_int,
        c_void_p, POINTER(c_int), c_void_p,
        c_void_p,
        POINTER(c_int), c_int,
        c_int,
        c_void_p, c_int,
    ]

    # OMP accessors
    _lib.soup_lib_omp_nselected.restype = c_int
    _lib.soup_lib_omp_nselected.argtypes = [c_void_p]
    _lib.soup_lib_omp_indices.restype = POINTER(c_int)
    _lib.soup_lib_omp_indices.argtypes = [c_void_p]
    _lib.soup_lib_omp_names.restype = c_void_p
    _lib.soup_lib_omp_names.argtypes = [c_void_p]
    _lib.soup_lib_omp_families.restype = c_void_p
    _lib.soup_lib_omp_families.argtypes = [c_void_p]
    _lib.soup_lib_omp_r2.restype = POINTER(c_double)
    _lib.soup_lib_omp_r2.argtypes = [c_void_p]
    _lib.soup_lib_omp_corr.restype = POINTER(c_double)
    _lib.soup_lib_omp_corr.argtypes = [c_void_p]
    _lib.soup_lib_free_omp.restype = None
    _lib.soup_lib_free_omp.argtypes = [c_void_p]


# ======================================================================
# Helper functions
# ======================================================================

def _pack_names(names_list):
    """Pack Python string list into C char[][128] array."""
    n = len(names_list)
    NameArray = (c_char * SOUP_MAX_NAME_LEN) * n
    arr = NameArray()
    for i, name in enumerate(names_list):
        b = name.encode('utf-8')[:SOUP_MAX_NAME_LEN - 1]
        arr[i][:len(b)] = b
        arr[i][len(b)] = b'\x00'
    return arr


def _extract_names(names_ptr, n):
    """Extract Python string list from C char(*)[128] pointer."""
    result = []
    for i in range(n):
        offset = i * SOUP_MAX_NAME_LEN
        raw = ctypes.string_at(names_ptr + offset, SOUP_MAX_NAME_LEN)
        result.append(raw.split(b'\x00', 1)[0].decode('utf-8'))
    return result


def _extract_families(fam_ptr, n):
    """Extract Python string list from C char(*)[32] pointer."""
    result = []
    for i in range(n):
        offset = i * SOUP_MAX_FAMILY_LEN
        raw = ctypes.string_at(fam_ptr + offset, SOUP_MAX_FAMILY_LEN)
        result.append(raw.split(b'\x00', 1)[0].decode('utf-8'))
    return result


# ======================================================================
# Public API
# ======================================================================

def build_lib_c(X_tr, X_te, feat_names, config=None,
                filter_threshold=0.005, dedup_threshold=0.95):
    """Build stages 0-2 soup library using C, with filter + dedup.

    Parameters
    ----------
    X_tr : array of shape (n_train, n_features)
    X_te : array of shape (n_test, n_features)
    feat_names : list of str
    config : SoupConfig or None
    filter_threshold : float
    dedup_threshold : float

    Returns
    -------
    PrimitiveLibrary with X (train) and X_val (test)
    """
    if config is not None:
        raise ValueError("build_lib_c uses native defaults; Python Config is unsupported")
    if not HAS_C_SOUP:
        raise RuntimeError("libsoup not found; set SIGNALFAULT_SOUP_LIB to an externally built library")

    X_tr = np.ascontiguousarray(X_tr, dtype=np.float64)
    X_te = np.ascontiguousarray(X_te, dtype=np.float64)
    n_tr, n_feat = X_tr.shape
    n_te = X_te.shape[0]

    numeric_mask = np.ones(n_feat, dtype=np.int32)
    c_names = _pack_names(feat_names)
    stats_ptr = c_void_p(0)

    m_ptr = _lib.soup_lib_build_and_clean(
        X_tr.ctypes.data_as(POINTER(c_double)),
        c_int(n_tr), c_int(n_feat),
        ctypes.cast(c_names, c_void_p),
        numeric_mask.ctypes.data_as(POINTER(c_int)),
        None,
        c_double(filter_threshold), c_double(dedup_threshold),
        ctypes.byref(stats_ptr),
    )
    if not m_ptr:
        raise RuntimeError("C soup build failed")

    ncols = _lib.soup_lib_ncols(m_ptr)
    data_ptr = _lib.soup_lib_data(m_ptr)
    names_ptr = _lib.soup_lib_names(m_ptr)
    fam_ptr = _lib.soup_lib_families(m_ptr)

    c_data = np.ctypeslib.as_array(data_ptr, shape=(ncols * n_tr,))
    X_train = c_data.reshape((ncols, n_tr), order='C').T.copy()

    c_names_out = _extract_names(names_ptr, ncols)
    c_fams_out = _extract_families(fam_ptr, ncols)

    c_keep = _pack_names(c_names_out)
    v_ptr = _lib.soup_lib_build_val(
        X_te.ctypes.data_as(POINTER(c_double)),
        c_int(n_te), c_int(n_feat),
        ctypes.cast(_pack_names(feat_names), c_void_p),
        numeric_mask.ctypes.data_as(POINTER(c_int)),
        None,
        stats_ptr,
        ctypes.cast(c_keep, c_void_p),
        c_int(ncols),
    )

    X_val = None
    if v_ptr:
        v_ncols = _lib.soup_lib_ncols(v_ptr)
        v_data_ptr = _lib.soup_lib_data(v_ptr)
        v_data = np.ctypeslib.as_array(v_data_ptr, shape=(v_ncols * n_te,))
        X_val = v_data.reshape((v_ncols, n_te), order='C').T.copy()
        _lib.soup_lib_free_matrix(v_ptr)

    _lib.soup_lib_free_matrix(m_ptr)
    _lib.soup_lib_free_stats(stats_ptr)

    return PrimitiveLibrary(X_train, c_names_out, c_fams_out, X_val=X_val)


def build_lib_deep_c(X_tr, X_te, feat_names, top_idx=None,
                     include_stage4=0,
                     filter_threshold=0.005, dedup_threshold=0.95):
    """Build stages 0-3 (FM/PM) soup library using C.

    Parameters
    ----------
    X_tr : array of shape (n_train, n_features)
    X_te : array of shape (n_test, n_features)
    feat_names : list of str
    top_idx : array of int, optional
    include_stage4 : int
        1 to include FM triangle + FM pulse.
    filter_threshold : float
    dedup_threshold : float

    Returns
    -------
    PrimitiveLibrary with X (train) and X_val (test)
    """
    if not HAS_C_SOUP:
        raise RuntimeError("libsoup not found; set SIGNALFAULT_SOUP_LIB to an externally built library")

    X_tr = np.ascontiguousarray(X_tr, dtype=np.float64)
    X_te = np.ascontiguousarray(X_te, dtype=np.float64)
    n_tr, n_feat = X_tr.shape
    n_te = X_te.shape[0]

    numeric_mask = np.ones(n_feat, dtype=np.int32)
    c_names = _pack_names(feat_names)
    stats_ptr = c_void_p(0)

    if top_idx is not None:
        top_idx = np.ascontiguousarray(top_idx, dtype=np.int32)
        c_top = top_idx.ctypes.data_as(POINTER(c_int))
        n_top = len(top_idx)
    else:
        c_top = None
        n_top = 0

    m_ptr = _lib.soup_lib_build_deep(
        X_tr.ctypes.data_as(POINTER(c_double)),
        c_int(n_tr), c_int(n_feat),
        ctypes.cast(c_names, c_void_p),
        numeric_mask.ctypes.data_as(POINTER(c_int)),
        None,
        c_top, c_int(n_top),
        c_int(include_stage4),
        c_double(filter_threshold), c_double(dedup_threshold),
        ctypes.byref(stats_ptr),
    )
    if not m_ptr:
        raise RuntimeError("C soup deep build failed")

    ncols = _lib.soup_lib_ncols(m_ptr)
    data_ptr = _lib.soup_lib_data(m_ptr)
    names_ptr = _lib.soup_lib_names(m_ptr)
    fam_ptr = _lib.soup_lib_families(m_ptr)

    c_data = np.ctypeslib.as_array(data_ptr, shape=(ncols * n_tr,))
    X_train = c_data.reshape((ncols, n_tr), order='C').T.copy()

    c_names_out = _extract_names(names_ptr, ncols)
    c_fams_out = _extract_families(fam_ptr, ncols)

    c_keep = _pack_names(c_names_out)

    if top_idx is not None:
        c_top_val = top_idx.ctypes.data_as(POINTER(c_int))
        n_top_val = len(top_idx)
    else:
        c_top_val = None
        n_top_val = 0

    v_ptr = _lib.soup_lib_build_deep_val(
        X_te.ctypes.data_as(POINTER(c_double)),
        c_int(n_te), c_int(n_feat),
        ctypes.cast(_pack_names(feat_names), c_void_p),
        numeric_mask.ctypes.data_as(POINTER(c_int)),
        None,
        stats_ptr,
        c_top_val, c_int(n_top_val),
        c_int(include_stage4),
        ctypes.cast(c_keep, c_void_p),
        c_int(ncols),
    )

    X_val = None
    if v_ptr:
        v_ncols = _lib.soup_lib_ncols(v_ptr)
        v_data_ptr = _lib.soup_lib_data(v_ptr)
        v_data = np.ctypeslib.as_array(v_data_ptr, shape=(v_ncols * n_te,))
        X_val = v_data.reshape((v_ncols, n_te), order='C').T.copy()
        _lib.soup_lib_free_matrix(v_ptr)

    _lib.soup_lib_free_matrix(m_ptr)
    _lib.soup_lib_free_stats(stats_ptr)

    return PrimitiveLibrary(X_train, c_names_out, c_fams_out, X_val=X_val)


def c_omp_select(library, y, max_features=20):
    """Compatibility wrapper for Python OMP; no native binary is required.

    Native matrix construction remains available separately via build_lib_c.
    """
    return residual_select(library, y, max_features=max_features, verbose=False)

"""Structural deduplication via operation fingerprints.

Catches mathematically identical features (e.g. a*b == b*a) before any
numeric computation, using frozenset-based fingerprints for commutative
ops and tuples for ordered ops.  This is an O(1) set lookup per feature,
eliminating duplicates before the expensive correlation-based dedup.

Fingerprint conventions:
    Commutative ops   -> frozenset of operand indices
    Ordered ops       -> tuple of operand indices
    Float parameters  -> rounded to 6 decimal places
"""


class StructuralDedup:
    """O(1) set-based dedup by mathematical identity.

    Usage::

        dedup = StructuralDedup()
        dedup.seed_raw(n_input)  # register input feature identities

        # In a builder loop:
        fp = ('prod', frozenset({i, j}))
        if dedup.is_new(fp, family='am_product'):
            # compute and append feature
            ...

        print(dedup.report())
    """

    def __init__(self):
        self._seen = set()
        self._skipped = 0
        self._skipped_by_family = {}

    def is_new(self, fingerprint, family=None):
        """Check if fingerprint is unseen.  Returns True if new, False if duplicate.

        Parameters
        ----------
        fingerprint : hashable
            Operation fingerprint (tuple/frozenset based).
        family : str, optional
            Family label for per-family skip tracking.
        """
        if fingerprint in self._seen:
            self._skipped += 1
            if family:
                self._skipped_by_family[family] = (
                    self._skipped_by_family.get(family, 0) + 1
                )
            return False
        self._seen.add(fingerprint)
        return True

    def seed_raw(self, n):
        """Register identity fingerprints for n input features.

        Call this when include_input=True so that derived features
        matching an input identity are caught.

        Parameters
        ----------
        n : int
            Number of input features.
        """
        for i in range(n):
            self._seen.add(('raw', (i,)))

    def report(self):
        """Return dedup statistics.

        Returns
        -------
        dict with keys:
            n_seen : int       -- total unique fingerprints registered
            n_skipped : int    -- total duplicates caught
            by_family : dict   -- family -> skip count
        """
        return {
            'n_seen': len(self._seen),
            'n_skipped': self._skipped,
            'by_family': dict(self._skipped_by_family),
        }

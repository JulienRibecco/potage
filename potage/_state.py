"""Transactional stage state and opaque rewind tokens."""
from dataclasses import dataclass
from functools import wraps


@dataclass(frozen=True)
class Checkpoint:
    length: int
    candidates: int
    memory: int
    families: frozenset
    tip: object


def atomic_stage(fn):
    """Restore stage bookkeeping if generation or selection raises."""
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        cp = self.checkpoint()
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            self.rewind(cp)
            raise
    return wrapped


class StateMixin:
    def checkpoint(self):
        """Return an opaque token for rewinding this pipeline's stage state."""
        return Checkpoint(len(self._stages), self._candidates_used,
                          self._memory_high_water,
                          frozenset(self._candidate_families_seen),
                          self._stages[-1] if self._stages else None)

    def rewind(self, cp=None):
        """Restore budget, family census and handles; no argument undoes one stage."""
        if cp is None:
            if not self._stages:
                raise ValueError("No stages to rewind")
            cp = self._checkpoints[-2] if len(self._checkpoints) > 1 else Checkpoint(0, 0, 0, frozenset(), None)
        if not isinstance(cp, Checkpoint):
            raise ValueError("Expected a token returned by checkpoint()")
        if cp.length > len(self._stages) or (cp.length and self._stages[cp.length - 1] is not cp.tip):
            raise ValueError("Checkpoint does not belong to the current pipeline history")
        del self._stages[cp.length:]
        del self._checkpoints[cp.length:]
        self._candidates_used = cp.candidates
        self._memory_high_water = cp.memory
        self._candidate_families_seen = set(cp.families)
        self._fs_to_stage = {id(s.output): i for i, s in enumerate(self._stages)}

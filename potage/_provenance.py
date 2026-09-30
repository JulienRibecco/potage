"""String-compatible feature labels with exact input-column ancestry.

Labels remain ordinary strings for comparison, display and serialization to
JSON. Python pickle additionally preserves ancestry for fitted pipelines.
"""


class FeatureName(str):
    def __new__(cls, value, sources):
        obj = super().__new__(cls, value)
        obj.sources = frozenset(sources)
        return obj

    def __reduce__(self):
        return (type(self), (str(self), self.sources))


def input_names(names):
    return [n if isinstance(n, FeatureName) else FeatureName(n, [n]) for n in names]


def derived_name(label, *operands):
    sources = frozenset().union(*(op.sources for op in operands if isinstance(op, FeatureName)))
    return FeatureName(label, sources)


def joined_name(separator, names):
    names = list(names)
    return derived_name(separator.join(names), *names)

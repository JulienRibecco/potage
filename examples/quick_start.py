"""Small offline demonstration; run from a checkout with Potage installed."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from potage import SoupPipe


def run():
    rng = np.random.RandomState(42)
    X = rng.uniform(-2, 2, size=(500, 3))
    y = X[:, 0] ** 2 + 2 * np.sin(2 * X[:, 1]) + rng.normal(0, .2, len(X))
    dev, test = train_test_split(np.arange(len(X)), test_size=.2, random_state=42)
    train, val = train_test_split(dev, test_size=.25, random_state=43)

    pipe = SoupPipe(X[train], y[train], ['a', 'b', 'c'],
                    X_val=X[val], y_val=y[val])
    raw = pipe.raw(select=6, method='greedy')
    carriers = pipe.carriers(raw, select=6, method='greedy')
    merged = pipe.fuse(raw, carriers)
    selected = pipe.select(merged, select=8, method='greedy')

    baseline = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(X[train], y[train])
    model = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(selected.X, y[train])
    print(f'Raw ridge test R²: {r2_score(y[test], baseline.predict(X[test])):.3f}')
    print(f'Potage + ridge test R²: {r2_score(y[test], model.predict(pipe.transform(X[test]))):.3f}')
    print('Selected:', ', '.join(selected.names))
    return pipe, model


if __name__ == '__main__':
    run()

"""Toy mean-aggregation logistic probe (Stage 0b pipeline validation).

A minimal FittableMeasurement realization: fit once per seed on a designated
split, freeze (weights hashed), score = w·mean_tokens(resid) + b with higher =
more deceptive. Logistic regression via L-BFGS on L2-regularized log-loss
(numpy/scipy only — no sklearn dependency at this stage).

NOT the Stage-C probe implementation; exists to exercise the
activation → probe → metrics pipeline end to end.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class FrozenProbe:
    w: np.ndarray  # (d_model,)
    b: float
    read_layer: int
    seed: int
    train_mean: np.ndarray
    train_std: np.ndarray

    @property
    def artifact_hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.w, self.train_mean, self.train_std):
            h.update(np.asarray(arr, dtype="<f8").tobytes())
        h.update(f"{self.b!r}|{self.read_layer}|{self.seed}".encode())
        return h.hexdigest()

    def score(self, mean_activation: np.ndarray) -> float:
        """Higher = more deceptive."""
        z = (mean_activation - self.train_mean) / self.train_std
        return float(z @ self.w + self.b)


class MeanAggLogisticProbe:
    name = "probe_mean_logistic"

    def __init__(self, read_layer: int, reg_coeff: float = 10.0):
        self.read_layer = read_layer
        self.reg_coeff = reg_coeff

    def fit(self, mean_acts: np.ndarray, labels: np.ndarray, *, seed: int) -> FrozenProbe:
        """mean_acts: (n, d_model); labels: 1 = deceptive, 0 = honest."""
        rng = np.random.default_rng(seed)
        mu, sd = mean_acts.mean(0), mean_acts.std(0) + 1e-6
        x = (mean_acts - mu) / sd
        y = labels.astype(np.float64)
        d = x.shape[1]
        w0 = np.concatenate([rng.normal(scale=1e-3, size=d), [0.0]])

        def loss(params):
            w, b = params[:-1], params[-1]
            z = x @ w + b
            # stable log(1 + exp(-|z|)) formulation
            ll = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
            grad_z = 1.0 / (1.0 + np.exp(-z)) - y
            grad_w = x.T @ grad_z + self.reg_coeff * w
            grad_b = grad_z.sum()
            return ll.sum() + 0.5 * self.reg_coeff * w @ w, np.concatenate([grad_w, [grad_b]])

        res = minimize(loss, w0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        w, b = res.x[:-1], float(res.x[-1])
        return FrozenProbe(w=w, b=b, read_layer=self.read_layer, seed=seed,
                           train_mean=mu, train_std=sd)

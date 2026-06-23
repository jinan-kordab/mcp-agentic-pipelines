"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrainingNode:
    beta: np.ndarray
    response: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    predicted_value: float
    confidence: float
    contributing_nodes: List[Dict]
    metadata: Dict = field(default_factory=dict)


class RadialInterpolPredictor:
    """RBF closed-form predictor. f_pred(x) = Σ ω_k(x)·f(β_k)·exp[-τ·K(x,β_k)] with Σ ω_k·exp[-τ·K] = 1."""

    def __init__(self, tau: float = 500.0, gamma: float = 1.0) -> None:
        self.tau = tau
        self.gamma = gamma
        self.nodes: List[TrainingNode] = []
        self.n: int = 0
        self.m: int = 0
        self._X_min: Optional[np.ndarray] = None
        self._X_max: Optional[np.ndarray] = None
        self._node_access_counts: np.ndarray = np.array([])
        self._total_predictions: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, metadata: Optional[List[Dict]] = None) -> None:
        self._X_min = X.min(axis=0)
        self._X_max = X.max(axis=0)
        X_t = self._phi_transform(X)
        meta_list = metadata or [{}] * len(X)
        self.nodes = [TrainingNode(beta=X_t[i], response=y[i], metadata=meta_list[i]) for i in range(len(X))]
        self.n = len(self.nodes)
        self.m = X.shape[1]
        self._node_access_counts = np.zeros(self.n)

    def predict(self, x: np.ndarray, top_k: int = 20,
                trace=None) -> Tuple[float, List[Dict]]:
        if self.n == 0:
            return 0.0, []
        x_t = self._phi_transform(x.reshape(1, -1))[0]
        contributions, w_sum, w_norm = [], 0.0, 0.0
        for i, node in enumerate(self.nodes):
            K = self._kernel(x_t, node.beta)
            w = np.exp(-self.tau * K)
            if w > 1e-15:
                w_sum += w * node.response
                w_norm += w
                self._node_access_counts[i] += 1
                contributions.append({"node_idx": i, "weight": float(w), "kernel_distance": float(K),
                                       "response": float(node.response), "metadata": node.metadata})
        if w_norm == 0:
            return 0.0, []
        f_pred = w_sum / w_norm
        self._total_predictions += 1
        contributions.sort(key=lambda c: c["weight"], reverse=True)
        if trace:
            trace.event(type("TE", (), {"value": "decision.prediction"})(), agent_name="RadialInterpol",
                        message=f"Predicted {f_pred:.4f} from {len(contributions)} active nodes",
                        data={"top_weight": contributions[0]["weight"] if contributions else 0})
        return f_pred, contributions[:top_k]

    def _kernel(self, x: np.ndarray, beta: np.ndarray) -> float:
        return self.gamma * float(np.dot(x - beta, x - beta))

    def _phi_transform(self, X: np.ndarray) -> np.ndarray:
        if self._X_min is None or self._X_max is None:
            return X
        denom = self._X_max - self._X_min
        denom[denom == 0] = 1.0
        return (X - self._X_min) / denom

    def auto_distill(self, min_access: int = 0) -> int:
        if self._total_predictions == 0:
            return 0
        keep = self._node_access_counts >= min_access
        removed = self.n - int(np.sum(keep))
        self.nodes = [n for i, n in enumerate(self.nodes) if keep[i]]
        self._node_access_counts = self._node_access_counts[keep]
        self.n = len(self.nodes)
        return removed

    def get_weights_distribution(self, x: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return np.array([])
        x_t = self._phi_transform(x.reshape(1, -1))[0]
        weights = np.array([np.exp(-self.tau * self._kernel(x_t, n.beta)) for n in self.nodes])
        s = weights.sum()
        return weights / s if s > 0 else weights

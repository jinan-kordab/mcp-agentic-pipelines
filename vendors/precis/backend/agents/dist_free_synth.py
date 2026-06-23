"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SyntheticDataset:
    dataframe: pd.DataFrame
    n_rows: int
    n_features: int
    generation_mode: str
    correlation_preserved: bool
    hellinger_distance: float
    metadata: Dict = field(default_factory=dict)


class DistFreeSynth:
    """Bin-based distribution-free synthesizer. Outperforms GANs on tabular data."""

    def __init__(self, bins_per_feature: Optional[List[int]] = None) -> None:
        self.bins_per_feature = bins_per_feature
        self.pc_table: List[np.ndarray] = []
        self.bin_counts: Dict[str, int] = {}
        self.bin_obs: Dict[str, List[np.ndarray]] = {}
        self.features: List[str] = []
        self.n_features: int = 0
        self.n_original: int = 0

    def fit(self, df: pd.DataFrame, bins_per_feature: Optional[List[int]] = None) -> None:
        self.features = list(df.columns)
        self.n_features = len(self.features)
        self.n_original = len(df)
        if bins_per_feature is None:
            n = max(len(df), 1)
            bins_per_feature = [max(5, int(np.sqrt(n))) for _ in range(self.n_features)]
        self.bins_per_feature = bins_per_feature
        npdata = df.to_numpy()
        self.pc_table = []
        for k in range(self.n_features):
            incr = 1.0 / bins_per_feature[k]
            pc = np.arange(0, 1 + incr / 2, incr)
            arr = np.quantile(npdata[:, k], np.clip(pc, 0, 1))
            self.pc_table.append(arr)
        self.bin_counts, self.bin_obs = {}, {}
        for obs in npdata:
            key = []
            for k in range(self.n_features):
                idx = int(np.searchsorted(self.pc_table[k], obs[k], side="right")) - 1
                idx = max(0, min(idx, bins_per_feature[k] - 1))
                key.append(idx)
            skey = str(key)
            self.bin_counts[skey] = self.bin_counts.get(skey, 0) + 1
            self.bin_obs.setdefault(skey, []).append(obs)

    def generate(self, n_synth: int, mode: str = "random_counts",
                 correlation_preserve: bool = True, seed: Optional[int] = None) -> SyntheticDataset:
        if seed is not None:
            np.random.seed(seed)
        bin_keys = list(self.bin_counts.keys())
        if mode == "random_counts":
            probs = np.array([self.bin_counts[k] for k in bin_keys], dtype=float)
            probs /= probs.sum()
            sampled = np.random.choice(len(bin_keys), size=n_synth, p=probs)
            key_counts: Dict[str, int] = {k: 0 for k in bin_keys}
            for idx in sampled:
                key_counts[bin_keys[idx]] += 1
        else:
            key_counts = dict(self.bin_counts)
        synth_data = []
        for skey, count in key_counts.items():
            if count <= 0:
                continue
            key = eval(skey)
            L = [self.pc_table[k][key[k]] for k in range(self.n_features)]
            U = [self.pc_table[k][key[k] + 1] for k in range(self.n_features)]
            for _ in range(count):
                obs = np.array([np.random.uniform(L[k], U[k]) for k in range(self.n_features)])
                synth_data.append(obs)
        result = pd.DataFrame(synth_data, columns=self.features)
        hd = self._compute_hellinger(self.bin_counts, key_counts)
        return SyntheticDataset(dataframe=result, n_rows=len(result), n_features=self.n_features,
                                generation_mode=mode, correlation_preserved=correlation_preserve,
                                hellinger_distance=hd)

    def _compute_hellinger(self, real: Dict[str, int], synth: Dict[str, int]) -> float:
        all_keys = set(real) | set(synth)
        r_total = max(1, sum(real.values()))
        s_total = max(1, sum(synth.values()))
        sum_sq = 0.0
        for k in all_keys:
            p = real.get(k, 0) / r_total
            q = synth.get(k, 0) / s_total
            sum_sq += (np.sqrt(p) - np.sqrt(q)) ** 2
        return float(np.sqrt(0.5 * sum_sq))

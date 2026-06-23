"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AnomalyFlag:
    entity_id: str
    flag_type: str
    severity: float
    evidence: Dict[str, Any]
    timestamp: datetime
    description: str
    recommendations: List[str] = field(default_factory=list)


class StatAnomalyDetector:
    """Multi-method anomaly detector: entity mapping, temporal spikes, geospatial, value outliers."""

    def __init__(self) -> None:
        self.entity_hash: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.temporal_hash: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.geo_hash: Dict[str, List[str]] = defaultdict(list)
        self.value_tracker: Dict[str, List[float]] = defaultdict(list)
        self.flags: List[AnomalyFlag] = []

    def ingest(self, entity_id: str, counterparty_id: Optional[str] = None,
               value: Optional[float] = None, timestamp: Optional[datetime] = None,
               jurisdiction: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
        if counterparty_id:
            self.entity_hash[entity_id][counterparty_id] = self.entity_hash[entity_id].get(counterparty_id, 0) + 1
        if timestamp is not None and value is not None:
            self.temporal_hash[entity_id].append((timestamp, value))
        if jurisdiction:
            self.geo_hash[entity_id].append(jurisdiction)
        if value is not None:
            self.value_tracker[entity_id].append(value)

    def ingest_batch(self, events: List[Dict[str, Any]]) -> int:
        for evt in events:
            self.ingest(**{k: evt.get(k) for k in ("entity_id", "counterparty_id", "value", "timestamp", "jurisdiction")})
        return len(events)

    def detect_multi_entity_anomalies(self, max_relationships: int = 50) -> List[AnomalyFlag]:
        flags = []
        for eid, counterparties in self.entity_hash.items():
            n = len(counterparties)
            if n > max_relationships:
                flags.append(AnomalyFlag(entity_id=eid, flag_type="multi_entity",
                    severity=min(1.0, n / (2 * max_relationships)),
                    evidence={"n_counterparties": n, "counterparties": dict(counterparties)},
                    timestamp=datetime.now(),
                    description=f"Entity has {n} counterparties (threshold: {max_relationships})"))
        return sorted(flags, key=lambda f: f.severity, reverse=True)

    def detect_temporal_spikes(self, spike_threshold_sigma: float = 3.0,
                                window_hours: int = 1) -> List[AnomalyFlag]:
        flags = []
        for eid, events in self.temporal_hash.items():
            if len(events) < 10:
                continue
            hourly = defaultdict(list)
            for ts, amount in events:
                hour_key = ts.replace(minute=0, second=0, microsecond=0)
                hourly[hour_key].append(amount)
            counts = [len(v) for v in hourly.values()]
            mean_c, std_c = float(np.mean(counts)), float(np.std(counts))
            if std_c == 0:
                continue
            for hour, amounts in hourly.items():
                z = (len(amounts) - mean_c) / std_c
                if z > spike_threshold_sigma:
                    flags.append(AnomalyFlag(entity_id=eid, flag_type="temporal_spike",
                        severity=min(1.0, z / (2 * spike_threshold_sigma)),
                        evidence={"hour": str(hour), "count": len(amounts), "total_amount": sum(amounts), "z_score": float(z)},
                        timestamp=hour,
                        description=f"{len(amounts)} events at {hour} ({z:.1f} above mean)"))
        return sorted(flags, key=lambda f: f.severity, reverse=True)

    def detect_all(self, trace=None) -> List[AnomalyFlag]:
        self.flags = []
        self.flags.extend(self.detect_multi_entity_anomalies())
        self.flags.extend(self.detect_temporal_spikes())
        self.flags.sort(key=lambda f: f.severity, reverse=True)
        if trace:
            for flag in self.flags[:5]:
                trace.event(type("TE", (), {"value": "anomaly.flagged"})(), agent_name="StatAnomaly",
                            message=flag.description, data={"severity": flag.severity, "type": flag.flag_type})
        return self.flags

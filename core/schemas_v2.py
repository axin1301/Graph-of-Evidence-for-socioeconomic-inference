from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class EvidenceNode:
    id: str
    modality: str
    observation: str
    implication: Dict[str, float]
    confidence: float
    source: Any = None

    # normalized optional fields
    source_type: str = "unknown"      # e.g. tool / agent / manual / verifier
    source_name: str = "unknown"
    region_scope: Optional[str] = None
    semantic_type: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ClaimNode:
    id: str
    task: str
    hypothesis: str
    estimate: float
    confidence: float
    support_eids: List[str] = field(default_factory=list)
    contradict_eids: List[str] = field(default_factory=list)
    status: str = "provisional"

    def to_dict(self):
        return asdict(self)


@dataclass
class TaskNode:
    id: str
    task_name: str
    target_field: str
    normalized_range: List[float]

    def to_dict(self):
        return asdict(self)


@dataclass
class SourceNode:
    id: str
    source_name: str
    source_type: str = "unknown"

    def to_dict(self):
        return asdict(self)


@dataclass
class RelationEdge:
    id: str
    from_id: str
    to_id: str
    relation_type: str
    observed: bool = True
    weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class EdgePotential:
    edge_id: str
    support_strength: float = 0.0
    conflict_strength: float = 0.0
    redundancy_strength: float = 0.0
    source_reliability: float = 0.5
    implication_alignment: float = 0.0
    potential_score: float = 0.0
    rationale: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
    
@dataclass
class FactorNode:
    id: str
    factor_name: str
    description: str
    task_relevance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


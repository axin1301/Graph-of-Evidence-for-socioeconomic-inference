from typing import List, Optional, Dict, Any
from core.schemas_v2 import (
    EvidenceNode, ClaimNode, TaskNode, SourceNode, RelationEdge
)
from graph.evidence_aggregator import aggregate_evidence
from graph.trace_processor import extract_trace_features

def normalize_source(source):
    """
    Convert raw source into (source_type, source_name).
    """
    if isinstance(source, dict):
        return source.get("type", "unknown"), source.get("name", "unknown")
    if isinstance(source, str):
        return "unknown", source
    return "unknown", "unknown"


def build_base_goe(
    evidences,
    claim,
    initial_verification=None,
    final_verification=None,
    case_id=None,
    setting_name=None,
    task_spec=None,
    reasoning_trace=None,
    use_balanced_aggregation=True
):
    evidence_nodes = []
    source_nodes = {}
    relation_edges = []

    for e in evidences:
        ed = e.to_dict()
        source_type, source_name = normalize_source(ed.get("source"))

        ev = EvidenceNode(
            id=ed["eid"],
            modality=ed["modality"],
            observation=ed["observation"],
            implication=ed["implication"],
            confidence=float(ed["confidence"]),
            source=ed.get("source"),
            source_type=source_type,
            source_name=source_name
        )

        ev_dict = ev.to_dict()
        ev_dict["spatial_layout"] = ed.get("spatial_layout", {})
        ev_dict["key_elements"] = ed.get("key_elements", [])
        ev_dict["local_variation"] = ed.get("local_variation")
        ev_dict["coverage"] = ed.get("coverage")
        ev_dict["semantic_type"] = ed.get("semantic_type")
        ev_dict["scene_type"] = ed.get("scene_type")
        ev_dict["informativeness"] = ed.get("informativeness")
        evidence_nodes.append(ev_dict)

        sid = f"SRC::{source_name}"
        if sid not in source_nodes:
            source_nodes[sid] = SourceNode(
                id=sid,
                source_name=source_name,
                source_type=source_type
            )

        relation_edges.append(RelationEdge(
            id=f"edge::{ev.id}::produced_by::{sid}",
            from_id=ev.id,
            to_id=sid,
            relation_type="produced_by",
            observed=True
        ))

    cd = claim.to_dict()
    claim_node = ClaimNode(
        id=cd["cid"],
        task=cd["task"],
        hypothesis=cd["hypothesis"],
        estimate=float(cd["estimate"]),
        confidence=float(cd["confidence"]),
        support_eids=list(cd.get("support_eids", [])),
        contradict_eids=list(cd.get("contradict_eids", []))
    )

    for eid in claim_node.support_eids:
        relation_edges.append(RelationEdge(
            id=f"edge::{eid}::support::{claim_node.id}",
            from_id=eid,
            to_id=claim_node.id,
            relation_type="support",
            observed=True
        ))

    for eid in claim_node.contradict_eids:
        relation_edges.append(RelationEdge(
            id=f"edge::{eid}::contradict::{claim_node.id}",
            from_id=eid,
            to_id=claim_node.id,
            relation_type="contradict",
            observed=True
        ))

    task_node = None
    aggregation = None
    if task_spec is not None:
        task_node = TaskNode(
            id=f"TASK::{task_spec['target_field']}",
            task_name=task_spec["task_name"],
            target_field=task_spec["target_field"],
            normalized_range=list(task_spec["normalized_range"])
        )

        relation_edges.append(RelationEdge(
            id=f"edge::{claim_node.id}::targets::{task_node.id}",
            from_id=claim_node.id,
            to_id=task_node.id,
            relation_type="targets",
            observed=True
        ))

        aggregation = aggregate_evidence(
            
            evidence_nodes,
            task_spec["target_field"],
            use_balanced=use_balanced_aggregation
        )
    trace_features = extract_trace_features(reasoning_trace)

    goe = {
        "meta": {
            "case_id": case_id,
            "setting": setting_name,
            "target_field": None if task_spec is None else task_spec["target_field"],
            "graph_version": "goe_base_v2",
            "balanced_aggregation": use_balanced_aggregation
        },
        "task": None if task_node is None else task_node.to_dict(),
        
        "evidences": evidence_nodes,
        "claims": [claim_node.to_dict()],
        "sources": [s.to_dict() for s in source_nodes.values()],
        "relations": [r.to_dict() for r in relation_edges],
        "aggregation": aggregation,
        "verification": {
            "initial": initial_verification,
            "final": final_verification
        },
        "reasoning_trace": reasoning_trace,
        "goe_trace": trace_features,
        "factors": [],
    }

    return goe




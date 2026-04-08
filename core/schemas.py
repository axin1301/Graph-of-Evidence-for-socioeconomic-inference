class Evidence:
    def __init__(
        self,
        eid=None,
        id=None,
        modality=None,
        observation=None,
        implication=None,
        confidence=0.5,
        source=None,
        spatial_layout=None,
        key_elements=None,
        local_variation=None,
        coverage=None,
        semantic_type=None,
        scene_type=None,
        informativeness=None
    ):
        self.eid = eid if eid is not None else id
        self.modality = modality
        self.observation = observation
        self.implication = implication
        self.confidence = confidence
        self.source = source

        # new structured fields
        self.spatial_layout = spatial_layout if spatial_layout is not None else {}
        self.key_elements = key_elements if key_elements is not None else []
        self.local_variation = local_variation
        self.coverage = coverage
        self.semantic_type = semantic_type
        self.scene_type = scene_type
        self.informativeness = informativeness

    def to_dict(self):
        return {
            "eid": self.eid,
            "modality": self.modality,
            "observation": self.observation,
            "implication": self.implication,
            "confidence": self.confidence,
            "source": self.source,
            "spatial_layout": self.spatial_layout,
            "key_elements": self.key_elements,
            "local_variation": self.local_variation,
            "coverage": self.coverage,
            "semantic_type": self.semantic_type,
            "scene_type": self.scene_type,
            "informativeness": self.informativeness
        }

class Claim:
    def __init__(
        self,
        cid,
        task,
        hypothesis,
        estimate,
        confidence,
        support_eids,
        contradict_eids=None,
        level=None,
        level_score=None,
        scale_type=None,
        unit=None
    ):
        self.cid = cid
        self.task = task
        self.hypothesis = hypothesis
        self.estimate = estimate
        self.confidence = confidence
        self.support_eids = support_eids
        self.contradict_eids = contradict_eids if contradict_eids is not None else []

        self.level = level
        self.level_score = level_score
        self.scale_type = scale_type
        self.unit = unit

    def to_dict(self):
        return {
            "cid": self.cid,
            "task": self.task,
            "hypothesis": self.hypothesis,
            "estimate": self.estimate,
            "confidence": self.confidence,
            "support_eids": self.support_eids,
            "contradict_eids": self.contradict_eids,
            "level": self.level,
            "level_score": self.level_score,
            "scale_type": self.scale_type,
            "unit": self.unit
        }




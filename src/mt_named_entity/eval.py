import logging
from collections import Counter
from typing import Dict, List

from .align import NERAlignment
from .ner import NERMarker

log = logging.getLogger(__name__)

ALIGNED = "aligned"
UPPER_BOUND = "upper_bound"
DISTANCE = "dist"
MATCHES = "matches"
ALL_METRICS = [ALIGNED, UPPER_BOUND, DISTANCE, MATCHES]


def get_markers_stats(ner_markers: List[List[NERMarker]]) -> Counter:
    return Counter(marker.tag for line_markers in ner_markers for marker in line_markers)


def get_metrics(alignments: List[List[NERAlignment]], upper_bound_ner_alignments: int) -> Dict[str, float]:
    dists = [alignment.distance for line_alignment in alignments for alignment in line_alignment]
    exact_match = sum(
        [
            1
            for line_alignment in alignments
            for alignment in line_alignment
            if alignment.marker_1.named_entity == alignment.marker_2.named_entity
        ]
    )
    return {
        ALIGNED: len(dists),
        UPPER_BOUND: upper_bound_ner_alignments,
        DISTANCE: sum(dists),
        MATCHES: exact_match,
    }

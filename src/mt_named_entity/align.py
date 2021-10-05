import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pyjarowinkler import distance
from scipy.optimize import linear_sum_assignment

from mt_named_entity.filter import filter_same_number_of_entity_types

from .ner import NERMarker, NERTag

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NERAlignment:
    """Hold NER alignment results."""

    distance: Optional[float]
    marker_1: NERMarker
    marker_2: NERMarker

    def __str__(self) -> str:
        return f"{self.marker_1}-{self.marker_2}-{self.distance}"


def get_min_hun_distance(words1: List[str], words2: List[str]) -> Tuple[float, List[Tuple[int, int, float]]]:
    """Calculate a similarity score between all pairs of words."""
    values = []
    hits: List[Tuple[int, int, float]] = []
    min_dist = 0
    if len(words1) == 0 or len(words2) == 0:
        return min_dist, hits
    for i in range(len(words1)):
        w1 = words1[i]
        row = []
        for j in range(len(words2)):
            w2 = words2[j]
            # Jaro-Winkler distance (not similarity score)
            row.append(1 - distance.get_jaro_distance(w1, w2, winkler=True, scaling=0.1))
        values.append(row)
    # Calculate the best pairing based on the similarity score.
    row_ids, col_ids = linear_sum_assignment(values)
    row_ids = list(row_ids)
    col_ids = list(col_ids)
    # The best alignment
    hits = []
    valsum = 0
    for i in range(len(row_ids)):
        row_id = row_ids[i]
        col_id = col_ids[i]
        hits.append((row_id, col_id, values[row_id][col_id]))
        valsum += values[row_id][col_id]

    min_dist = valsum / (len(words1) + len(words2))

    return min_dist, hits


def align_markers_by_jaro_winkler(ner_markers_1: List[NERMarker], ner_markers_2: List[NERMarker]) -> List[NERAlignment]:
    """Aligns NERMarkers based on Jaro-Winkler distance."""
    try:
        min_dist, hits = get_min_hun_distance(
            [ner_marker.named_entity for ner_marker in ner_markers_1],
            [ner_marker.named_entity for ner_marker in ner_markers_2],
        )
    except (ValueError, distance.JaroDistanceException):
        log.exception(f"Bad NER markers: {ner_markers_1=}, {ner_markers_2}")
    return [NERAlignment(cost, ner_markers_1[hit_1], ner_markers_2[hit_2]) for hit_1, hit_2, cost in hits]


def align_markers_by_order(ner_markers_1: List[NERMarker], ner_markers_2: List[NERMarker]) -> List[NERAlignment]:
    """Aligns NERTags based on order."""
    ner_markers_1, ner_markers_2 = filter_same_number_of_entity_types(ner_markers_1, ner_markers_2) # type: ignore
    # We sort the NERMarkers by their tag.
    ner_markers_1 = sorted(ner_markers_1, key=lambda x: x.tag)
    ner_markers_2 = sorted(ner_markers_2, key=lambda x: x.tag)
    return [NERAlignment(None, marker_1, marker_2) for marker_1, marker_2 in zip(ner_markers_1, ner_markers_2)]

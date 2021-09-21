import logging
from typing import List, Optional

from islenska import Bin

from .align import NERAlignment, align_markers_by_order
from .ner import NERMarker, NERTag

log = logging.getLogger(__name__)


class Corrector:
    def __init__(self, should_correct_to_nomintaive_case=True) -> None:
        self.b = Bin()
        self.should_correct_icelandic_to_nominative_case = should_correct_to_nomintaive_case

    def __call__(self, src_text: str, tgt_text: str, alignment: NERAlignment) -> Optional[str]:
        """Return the corrected marker_2 in the alignment. If no change is applied, return None."""
        if self.should_correct_icelandic_to_nominative_case:
            return self.correct_icelandic_to_nominative_case(alignment.marker_1, alignment.marker_2)
        return None

    def correct_icelandic_to_nominative_case(self, src_marker: NERMarker, tgt_marker: NERMarker) -> Optional[str]:
        """Correct the tgt NERMarker to nominative case using BinPackage. Return None if no change"""
        m = self.b.lookup_variants(src_marker.named_entity, "no", ("NF"))
        if not m:
            return None
        if len(m) > 1:
            log.warning(f"Warning: multiple matches for {src_marker.named_entity}: {m}")
            return None
        return m[0].bmynd


def correct_line(
    src_line: str, tgt_line: str, src_entities: List[NERTag], tgt_entities: List[NERTag], corrector: Corrector
) -> str:
    """
    Return the corrected tgt_line. If no change is applied, return the original tgt_line.
    Corrects the tgt_line by removing wrong entities and replacing them with the correct ones as defined by the Corrector.
    """
    # First we create the NERMarkers for the source and target lines.
    src_markers = [NERMarker.from_tag(tag, src_line) for tag in src_entities]
    tgt_markers = [NERMarker.from_tag(tag, tgt_line) for tag in tgt_entities]
    # Then we align the NEs in the source and target by order.
    alignments = align_markers_by_order(src_markers, tgt_markers)
    # Then we correct the source and target lines by removing the wrong entities.
    corrected_alignments = []
    for alignment in alignments:
        correction = corrector(src_line, tgt_line, alignment)
        if correction is not None:
            corrected_alignments.append((correction, alignment))
    # We order the alignments by the start_idx of the target_markers, descending.
    corrected_alignments.sort(key=lambda x: x[1].marker_2.start_idx, reverse=True)
    # Then we correct the source and target lines by replacing the wrong entities with the correct ones.
    for correction, alignment in corrected_alignments:
        tgt_line = tgt_line[: alignment.marker_2.start_idx] + correction + tgt_line[alignment.marker_2.end_idx :]
    return tgt_line

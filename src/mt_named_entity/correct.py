import logging
from typing import Dict, List, Optional

from islenska import Bin
from islenska.bindb import KsnidList

from .align import align_markers_by_order
from .ner import NERMarker

log = logging.getLogger(__name__)


class Corrector:
    def __init__(self, should_correct_to_nomintaive_case: bool, corrections: Optional[Dict[str, str]] = None) -> None:
        self.b = Bin()
        self.should_correct_icelandic_to_nominative_case = should_correct_to_nomintaive_case
        self.corrections = corrections if corrections else {}

    def __call__(self, src_text: str, tgt_text: str, src_ne: str, tgt_ne: str) -> Optional[str]:
        """Return the corrected marker_2 in the alignment. If no change is applied, return None."""
        # First we check if the src_ne is in the corrections dict and if so, we return the value
        if src_ne in self.corrections:
            return self.corrections[src_ne]
        # Otherwise we check if we should attempt to inflect the marker_1 to nominative case.
        if self.should_correct_icelandic_to_nominative_case:
            return self.correct_icelandic_to_nominative_case(src_ne)
        return None

    def correct_icelandic_to_nominative_case(self, src_ne: str) -> Optional[str]:
        """Correct the tgt NERMarker to nominative case using BinPackage. Return None if no change"""
        # We split the src_ne into words by whitespace and attempt to correct each wordA
        corrected_parts = []
        for src_ne_part in src_ne.split(" "):
            corrected_part = self._inflect_using_bin(src_ne_part)
            if corrected_part:
                # If we have a correction, we return the corrected word
                corrected_parts.append(corrected_part)
            else:
                # Otherwise we add the original word
                corrected_parts.append(src_ne_part)
        return " ".join(corrected_parts)

    def _inflect_using_bin(self, src_ne: str) -> Optional[str]:
        """Inflect the src_ne using BinPackage. Return None if no change"""

        def get_by_best_einkunn(m: KsnidList) -> KsnidList:
            # 1 = generally accepted and modern, 0 accepted but old Icelandic, 2 = not fully accepted, etc.
            for einkunn in (1, 0, 2, 3, 4, 5):
                filtered = [match for match in m if match.einkunn == einkunn]
                if len(filtered) > 0:
                    return filtered
            raise ValueError(f"No einkunn available: {m}")

        def get_by_best_malsnid(m: KsnidList) -> KsnidList:
            # We prefer results with no malsnid.
            for malsnid in ("", "STAD", "GAM", "URE"):
                filtered = [match for match in m if match.malsnid == malsnid]
                if len(filtered) > 0:
                    return filtered
            raise ValueError(f"No malsnid available: {m}")

        def get_by_birting(m: KsnidList) -> KsnidList:
            # We prefer results with no birting.
            for birting in ("K", "V"):
                filtered = [match for match in m if match.birting == birting]
                if len(filtered) > 0:
                    return filtered
            raise ValueError(f"No birting available: {m}")

        m = self.b.lookup_variants(src_ne, "no", ("NF"))
        if not m:
            return None

        if len(m) == 1:
            return m[0].bmynd
        # We check if their forms actually differ.
        word_forms = set(match.bmynd for match in m)
        if len(word_forms) == 1:
            return word_forms.pop()
        # Their forms differ, so we need to do some heuristic filtering to selected the most "accepted one"
        m_best_einkunn = get_by_best_einkunn(m)
        if len(m_best_einkunn) == 1:
            # We have a single match with einkunn 1.
            return m_best_einkunn[0].bmynd
        m_best_birting = get_by_birting(m)
        if len(m_best_birting) == 1:
            # We have a single match with birting.
            return m_best_birting[0].bmynd
        m_best_malsnid = get_by_best_malsnid(m)
        if len(m_best_malsnid) == 1:
            # We have a single match with malsnid.
            return m_best_malsnid[0].bmynd
        raise ValueError(f"Multiple word forms for {src_ne}: {word_forms}, matches={m}")


def correct_line(
    src_line: str, tgt_line: str, src_markers: List[NERMarker], tgt_markers: List[NERMarker], corrector: Corrector
) -> str:
    """
    Return the corrected tgt_line. If no change is applied, return the original tgt_line.
    Corrects the tgt_line by removing wrong entities and replacing them with the correct ones as defined by the Corrector.
    """
    # Then we align the NEs in the source and target by order.
    alignments = align_markers_by_order(src_markers, tgt_markers)
    # Then we correct the source and target lines by removing the wrong entities.
    corrected_alignments = []
    for alignment in alignments:
        correction = corrector(src_line, tgt_line, alignment.marker_1.named_entity, alignment.marker_2.named_entity)
        if correction is not None:
            corrected_alignments.append((correction, alignment))
    # We order the alignments by the start_idx of the target_markers, descending.
    corrected_alignments.sort(key=lambda x: x[1].marker_2.start_idx, reverse=True)
    # Then we correct the source and target lines by replacing the wrong entities with the correct ones.
    for correction, alignment in corrected_alignments:
        tgt_line = tgt_line[: alignment.marker_2.start_idx] + correction + tgt_line[alignment.marker_2.end_idx :]
    return tgt_line

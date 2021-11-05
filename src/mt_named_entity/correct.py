import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

from islenska import Bin
from islenska.bindb import KsnidList

from .align import align_markers_by_order
from .ner import NERMarker

log = logging.getLogger(__name__)


class CorrectionResult(Enum):
    """
    Possible results of a correction.
    """

    NO_CORRECTION = 1
    AMBIGUOUS = 2
    WAS_CORRECT = 3
    CORRECTED = 4

    def merge(self, other):
        """Merges two CorrectionResults to a single CorrectionResult"""
        if self.value >= other.value:
            return self
        else:
            return other


class Corrector:
    """Applies corrections to NERMarkers and tracks statistics."""

    STATISTICS_DICTIONARY_KEY = "successful_dictionary_lookup"
    STATISTICS_NOMINATIVE_CASE = "nominative_case_inflections"

    def __init__(self, should_correct_to_nomintaive_case: bool, corrections: Optional[Dict[str, str]] = None) -> None:
        self.b = Bin()
        self.should_correct_icelandic_to_nominative_case = should_correct_to_nomintaive_case
        self.corrections = corrections if corrections else {}
        self.correction_statistics = {
            self.STATISTICS_DICTIONARY_KEY: {correction_result: 0 for correction_result in CorrectionResult},
            self.STATISTICS_NOMINATIVE_CASE: {correction_result: 0 for correction_result in CorrectionResult},
        }

    def __call__(
        self, src_text: str, tgt_text: str, src_ner_marker: NERMarker, tgt_ner_marker: NERMarker
    ) -> Tuple[str, CorrectionResult]:
        """Return the corrected marker_2 in the alignment. If no change is applied, return None."""
        # First we check if the src_ne is in the corrections dict and if so, we return the value
        if src_ner_marker.named_entity in self.corrections:
            correction = self.corrections[src_ner_marker.named_entity]
            correction_result = CorrectionResult.WAS_CORRECT
            if tgt_ner_marker.named_entity != correction:
                log.debug(f"Using corrections dictionary: {tgt_ner_marker.named_entity} -> {correction}")
                correction_result = CorrectionResult.CORRECTED
            self.correction_statistics[self.STATISTICS_DICTIONARY_KEY][correction_result] += 1
            return correction, correction_result
        # Otherwise we apply rules based on the tag
        if src_ner_marker.tag == "L":
            pass
        elif src_ner_marker.tag == "O":
            pass
        elif src_ner_marker.tag == "M":
            pass
        elif src_ner_marker.tag == "P":
            # Otherwise we check if we should attempt to inflect the source_marker to nominative case.
            if self.should_correct_icelandic_to_nominative_case:
                inflected_src_ne, correction_result = self.inflect_to_nominative_case(src_ner_marker.named_entity)
                # We check if we were successfully to inflect to nominative case.
                # This implies that the src_ne was an Icelandic Name.
                # If so, we want to make sure that it is inflected to nominative case in TGT.
                if correction_result == CorrectionResult.CORRECTED or correction_result == CorrectionResult.WAS_CORRECT:
                    # Was the tgt_ne already in nominative case?
                    if inflected_src_ne == tgt_ner_marker.named_entity:
                        correction_result = CorrectionResult.WAS_CORRECT
                        correction = tgt_ner_marker.named_entity
                        log.debug(
                            f"TGT was already nominative case. SRC={src_ner_marker.named_entity}, TGT={correction}."
                        )
                    # The tgt_ne was not in nominative case. Fix it!
                    else:
                        correction_result = CorrectionResult.CORRECTED
                        correction = inflected_src_ne
                        log.debug(
                            f"Inflecting to nominative case. SRC={src_ner_marker.named_entity}, \
TGT={tgt_ner_marker.named_entity} -> {correction}."
                        )
                # We were unable to inflect the src_ne to nominative case and are thus unable to validate tgt_ne.
                else:
                    correction = tgt_ner_marker.named_entity
                self.correction_statistics[self.STATISTICS_NOMINATIVE_CASE][correction_result] += 1
                return correction, correction_result
        else:
            raise ValueError(f"Unknown tag: {src_ner_marker.tag}")
        return tgt_ner_marker.named_entity, CorrectionResult.NO_CORRECTION

    def inflect_to_nominative_case(self, src_ne: str) -> Tuple[str, CorrectionResult]:
        """Inflect the src NE to nominative case using BinPackage.
        Return the src_ne and the inflection result.
        If no change is applied, return the src_ne and CorrectionResult.NO_CORRECTION.
        If the src_ne is already in nominative case, return the src_ne and CorrectionResult.WAS_CORRECT.
        If the src_ne was inflected, return the inflected src_ne and CorrectionResult.CORRECTED."""
        # We split the src_ne into words by whitespace and attempt to correct each word.
        original_src_ne = src_ne.split(" ")
        inflected_src_ne = [
            self._inflect_using_bin(src_ne_part, case="NF", assume_uppercase=True) for src_ne_part in original_src_ne
        ]
        # We merge the lists - using the original words if the inflected word is None.
        merged_correction_result = CorrectionResult.NO_CORRECTION
        result = []
        for src_ne_part, (inflected_src_ne_part, correction_result) in zip(original_src_ne, inflected_src_ne):
            merged_correction_result = merged_correction_result.merge(correction_result)
            result.append(inflected_src_ne_part if correction_result == CorrectionResult.CORRECTED else src_ne_part)

        return " ".join(result), merged_correction_result

    def _inflect_using_bin(self, word: str, case: str = "NF", assume_uppercase=True) -> Tuple[str, CorrectionResult]:
        """Inflect the word using BinPackage.
        If no change is applied, return the src_ne and CorrectionResult.NO_CORRECTION.
        If the src_ne is already in nominative case, return the src_ne and CorrectionResult.WAS_CORRECT.
        If the src_ne was inflected, return the inflected src_ne and CorrectionResult.CORRECTED."""

        def get_by_best_einkunn(m: KsnidList) -> KsnidList:
            # 1 = generally accepted and modern, 0 accepted but old Icelandic, 2 = not fully accepted, etc.
            for einkunn in (1, 0, 2, 3, 4, 5):
                filtered = [match for match in m if match.einkunn == einkunn]
                if len(filtered) > 0:
                    return filtered
            log.warning(f"No einkunn available: {m}")
            return m

        def get_by_best_hluti(m: KsnidList) -> KsnidList:
            # We prefer to use ism (person name) over föð (paternal name) over örn (place name) and then bær (town name)
            for hluti in ("ism", "föð", "örn", "bær"):
                filtered = [match for match in m if match.hluti == hluti]
                if len(filtered) > 0:
                    return filtered
            log.warning(f"No hluti available: {m}")
            return m

        def get_by_best_malsnid(m: KsnidList) -> KsnidList:
            # We prefer results with no malsnid.
            for malsnid in ("", "STAD", "GAM", "URE"):
                filtered = [match for match in m if match.malsnid == malsnid]
                if len(filtered) > 0:
                    return filtered
            log.warning(f"No malsnid available: {m}")
            return m

        def get_by_birting(m: KsnidList) -> KsnidList:
            # We prefer results with no birting.
            for birting in ("K", "V"):
                filtered = [match for match in m if match.birting == birting]
                if len(filtered) > 0:
                    return filtered
            log.warning(f"No birting available: {m}")
            return m

        def get_by_et(m: KsnidList) -> KsnidList:
            # We prefer results which are singular (ET).
            filtered = [match for match in m if "ET" in match.mark]
            if len(filtered) > 0:
                return filtered
            log.warning(f"No ET available: {m}")
            return m

        def return_result(original_word: str, inflected_word: str):
            if original_word == inflected_word:
                return original_word, CorrectionResult.WAS_CORRECT
            else:
                return inflected_word, CorrectionResult.CORRECTED

        m = self.b.lookup_variants(word, "no", (case))
        # Filter out results that do not start uppercased.
        if assume_uppercase:
            m = [match for match in m if match.bmynd[0].isupper()]
        if not m:
            return word, CorrectionResult.NO_CORRECTION
        # We check if their forms actually differ.
        word_forms = set(match.bmynd for match in m)
        if len(word_forms) == 1:
            return return_result(word, word_forms.pop())
        # Their forms differ, so we need to do some heuristic filtering to selected the most "accepted one"
        for f in [get_by_best_hluti, get_by_best_einkunn, get_by_birting, get_by_best_malsnid, get_by_et]:
            m = f(m)
            if len(m) == 1:
                return return_result(word, m[0].bmynd)
        # We have no suggestions.
        if len(m) > 1:
            log.warning(f"Multiple results for {word}: matches={m}")
            return word, CorrectionResult.AMBIGUOUS
        return word, CorrectionResult.NO_CORRECTION


def correct_line(
    src_line: str, tgt_line: str, src_markers: List[NERMarker], tgt_markers: List[NERMarker], corrector: Corrector
) -> Tuple[str, CorrectionResult]:
    """
    Return the corrected tgt_line and the correction result. If no change is applied, return the original tgt_line.
    Corrects the tgt_line by removing wrong entities and replacing them with the correct ones as defined by the Corrector.
    """
    final_correction_result = CorrectionResult.NO_CORRECTION
    # Then we align the NEs in the source and target by order.
    alignments = align_markers_by_order(src_markers, tgt_markers)
    # Then we correct the source and target lines by removing the wrong entities.
    corrections = [
        (alignment, corrector(src_line, tgt_line, alignment.marker_1, alignment.marker_2)) for alignment in alignments
    ]
    # We order the corrections by the alignments' start_idx of the target_markers, descending.
    # This allows us to correct the target line in the correct order.
    corrections.sort(key=lambda x: x[0].marker_2.start_idx, reverse=True)
    # Then we correct the target line by replacing the wrong entities with the correct ones.
    for alignment, (correction, correction_result) in corrections:
        tgt_line = tgt_line[: alignment.marker_2.start_idx] + correction + tgt_line[alignment.marker_2.end_idx :]
        final_correction_result.merge(correction_result)
    return tgt_line, final_correction_result

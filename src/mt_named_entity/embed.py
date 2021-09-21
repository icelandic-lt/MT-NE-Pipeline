import re
from typing import List, Match, Optional, Tuple

from mt_named_entity.ner import NERTag

NULL_TAG = "O"
# Uses BIO and all entities start with B.
PER = "P"
LOC = "L"
ORG = "O"
MISC = "M"
DATE = "D"
TIME = "T"
MON = "$"
PERC = "%"
TAGS = {PER, LOC, ORG, MISC, DATE, TIME, MON, PERC}
BIO_MAPPER = {NULL_TAG: NULL_TAG, "B": "B", "I": "I", "U": "B", "L": "I"}
IS_TAGS = {
    "Person": PER,
    "Location": LOC,
    "Organization": ORG,
    "Miscellaneous": MISC,
    "Date": DATE,
    "Time": TIME,
    "Money": MON,
    "Percent": PERC,
}
HF_TAGS = {
    "MISC": MISC,
    "PER": PER,
    "ORG": ORG,
    "LOC": LOC,
}
SP_TAGS = {
    "CARDINAL": MISC,
    "GPE": ORG,
    "ORG": ORG,
    "PERSON": PER,
    "DATE": DATE,
    "EVENT": MISC,
    "FAC": MISC,  # ?
    "LANGUAGE": MISC,
    "LAW": MISC,
    "LOC": LOC,
    "MONEY": MON,
    "NORP": MISC,
    "ORDINAL": MISC,
    "PERCENT": PERC,
    "PRODUCT": MISC,
    "QUANTITY": MISC,
    "TIME": TIME,
    "WORK_OF_ART": MISC,
}
TAG_MAPPER = {**IS_TAGS, **HF_TAGS, **SP_TAGS}

ENTITY_MARKERS_START = re.compile(f"<[{'|'.join(TAGS)}]+>")
ENTITY_MARKERS_END = re.compile(f"</[{'|'.join(TAGS)}]+>")

ENTITY_MARKERS = re.compile(f"</?[{'|'.join(TAGS)}]+>")

def embed_ner_tags(sentence: str, ner_tags: List[NERTag]) -> str:
    # Reverse the ner_tags, so we start at the end of the sentence.
    ner_tags = reversed(ner_tags) # type: ignore
    for ner_tag in ner_tags:
        entity = sentence[ner_tag.start_idx : ner_tag.end_idx]
        embedded_tag = TAG_MAPPER[ner_tag.tag]
        embedded_entity = f"<{embedded_tag}>{entity}</{embedded_tag}>"
        sentence = sentence[: ner_tag.start_idx] + embedded_entity + sentence[ner_tag.end_idx :]
    return sentence

def extract_ner_tags(sentence: str) -> Tuple[str, List[NERTag]]:
    """Extracts and removes the NER tags from the given sentence."""

    def find_start(sentence: str, pos: int) -> Optional[Match]:
        """Finds the start of an entity."""
        return ENTITY_MARKERS_START.search(sentence, pos)

    def find_end(sentence: str, pos: int) -> Optional[Match]:
        """Finds the end of an entity."""
        return ENTITY_MARKERS_END.search(sentence, pos)

    def extract_ner_tag(start_match: Match, end_match: Match, sentence: str) -> Tuple[str, NERTag]:
        """Extracts the NER tag from the given sentence."""
        start_idx = start_match.start()
        end_idx = end_match.end()
        start_tag = start_match.group()[1:-1]
        end_tag = end_match.group()[2:-1]
        if start_tag != end_tag:
            raise ValueError(f"Start tag {start_tag} and end tag {end_tag} do not match.")
        entity = sentence[start_match.end() : end_match.start()]
        corrected_end_idx = end_idx - 3 - 4 # -3 for the <X> and -4 for the </X>
        return (sentence[: start_idx] + entity + sentence[end_idx :], NERTag(start_tag, start_idx, corrected_end_idx))

    # Reverse the ner_tags, so we start at the end of the sentence.
    tags = []
    current_idx = 0
    # TODO: What if there is </X> first?
    try:
        while start_match := find_start(sentence, current_idx):
            current_idx = start_match.start()
            end_match = find_end(sentence, current_idx)
            if not end_match:
                raise ValueError(f"No end tag found for start tag {start_match.group()}.")
            sentence, tag = extract_ner_tag(start_match, end_match, sentence)
            tags.append(tag)
    except ValueError as e:
        print(e)
        print(sentence)
        print(tags)
    # TODO: Clean up all remaining <X> and </X> tags and correct the idxs of the tags based on the number of deletions.
    return sentence, tags
       


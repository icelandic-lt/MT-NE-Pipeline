import re
from typing import List

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

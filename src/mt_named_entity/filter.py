from typing import Counter, List

from mt_named_entity.ner import NERTag

PER = "P"
LOC = "L"
ORG = "O"
MISC = "M"
DATE = "D"
TIME = "T"
MON = "$"
PERC = "%"
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
ALLOWED_TAGS = {PER, LOC, ORG}

def filter_same_number_of_entity_types(src_NEs: List[NERTag], tgt_NEs: List[NERTag]) -> bool:
    """Filter translations by named entities types count. If the src and tgt do not contain the same number of NEs types it is rejected."""
    if len(src_NEs) != len(tgt_NEs):
        return False
    src_counter = Counter([tag.tag for tag in src_NEs])
    tgt_counter = Counter([tag.tag for tag in tgt_NEs])
    if src_counter != tgt_counter:
        return False
    return True

def map_named_entity_types(ner_tags: List[NERTag]) -> List[NERTag]:
    """Map named entities types. We map different system NE markers to a uniform format using TAG_MAPPER."""
    return [NERTag(TAG_MAPPER[tag.tag], tag.start_idx, tag.end_idx) for tag in ner_tags]


def filter_named_entity_types(ner_tags: List[NERTag]) -> List[NERTag]:
    """Filter named entities types. We only allow Organization, Location and Person."""
    return [tag for tag in ner_tags if tag.tag in ALLOWED_TAGS]

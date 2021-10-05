from typing import Counter, List, Tuple

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
ALL_TAGS = [PER, LOC, ORG, MISC, DATE, TIME, MON, PERC]
ALLOWED_TAGS = {PER, LOC, ORG}


def filter_same_number_of_entity_types(src_NEs: List[NERTag], tgt_NEs: List[NERTag]) -> Tuple[List[NERTag], List[NERTag]]:
    """Filter translations by named entities types count. If the src and tgt do not contain the same number of NEs types for some type, that type is filtered out."""
    src_counter = Counter([tag.tag for tag in src_NEs])
    tgt_counter = Counter([tag.tag for tag in tgt_NEs])
    # All the tags that are in both counters
    allowed_tags = set(src_counter.keys()) & set(tgt_counter.keys())
    # We then check the number of occurrences of each tag in the src and tgt
    src_NEs = [tag for tag in src_NEs if tag.tag in allowed_tags and src_counter[tag.tag] == tgt_counter[tag.tag]]
    tgt_NEs = [tag for tag in tgt_NEs if tag.tag in allowed_tags and src_counter[tag.tag] == tgt_counter[tag.tag]]
    return src_NEs, tgt_NEs

def map_named_entity_types(ner_tags: List[NERTag]) -> List[NERTag]:
    """Map named entities types. We map different system NE markers to a uniform format using TAG_MAPPER."""
    return [NERTag(TAG_MAPPER[tag.tag], tag.start_idx, tag.end_idx) for tag in ner_tags]


def filter_named_entity_types(ner_tags: List[NERTag]) -> List[NERTag]:
    """Filter named entities types. We only allow Organization, Location and Person."""
    return [tag for tag in ner_tags if tag.tag in ALLOWED_TAGS]

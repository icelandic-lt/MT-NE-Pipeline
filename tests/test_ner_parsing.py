from mt_named_entity.ner import IS_NER


def test_is_find_token():
    sent = "$ 1 Pick Three (7-4-1) greiddi $14,20. Pick Three Pool $12,109. $1 Consolation Pick Three (7-5-1) greiddi $7,10."
    tokens = "$1 Pick Three ( 7-4-1 ) greiddi $14,20 . Pick Three Pool $12,109 . $1 Consolation Pick Three ( 7-5-1 ) greiddi $7,10 .".split(" ")
    labels = "B-Organization B-Organization I-Organization O O O O B-Money O B-Organization I-Organization I-Organization B-Money O B-Organization B-Organization I-Organization I-Organization O O O O B-Money O".split(" ")
    # It should not crash
    tags = IS_NER.parse_ner_tags(sent, tokens, labels)
    assert True


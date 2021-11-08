from islenska import Bin

from mt_named_entity.cli import read_ner_tags, to_ner_markers
from mt_named_entity.correct import CorrectionResult, Corrector, correct_line
from mt_named_entity.embed import embed_ner_tags


def test_variant_lookup():
    b = Bin()
    m = b.lookup_variants("Einari", "no", ("NF"))
    assert m[0].bmynd == "Einar"

def test_corrector_multiple_tokens():
    corrector = Corrector(should_correct_to_nomintaive_case=True)
    result, correction_result = corrector.inflect_to_nominative_case("Einars Jónssonar")
    assert result == "Einar Jónsson"
    assert correction_result == CorrectionResult.CORRECTED


def test_corrector_hildur():
    corrector = Corrector(should_correct_to_nomintaive_case=True)
    result, correction_result = corrector.inflect_to_nominative_case("Hildar Sigurðardóttur")
    assert result == "Hildur Sigurðardóttir"
    assert correction_result == CorrectionResult.CORRECTED

def test_was_correct_hildur():
    corrector = Corrector(should_correct_to_nomintaive_case=True)
    result, correction_result = corrector.inflect_to_nominative_case("Hildur Sigurðardóttir")
    assert result == "Hildur Sigurðardóttir"
    assert correction_result == CorrectionResult.WAS_CORRECT

def test_corrector_hluti():
    corrector = Corrector(should_correct_to_nomintaive_case=True)
    result, _ = corrector._inflect_using_bin("Hólm")
    assert result == "Hólm"

def test_foreign_name():
    ne = "Johaug"
    corrector = Corrector(should_correct_to_nomintaive_case=True)
    result, correction_result = corrector.inflect_to_nominative_case(ne)
    assert result == ne
    assert correction_result == CorrectionResult.NO_CORRECTION

def test_correct_line_ulla():
    corrector = Corrector(should_correct_to_nomintaive_case=True)
    src_line = "Úlla Árdal er með minnstu starfsreynsluna af þessum þremur en hún kom til starfa á RÚV árið 2019."
    src_ner_tags = "P:0:10 O:83:86"
    tgt_line = "Úlla Árdal has the least work experience of the three, who came to work on RÚV in 2019."
    tgt_ner_tags = "P:0:10 O:75:78"
    ref_markers = to_ner_markers(read_ner_tags([src_ner_tags]), [src_line])
    sys_markers = to_ner_markers(read_ner_tags([tgt_ner_tags]), [tgt_line])

    result = correct_line(src_line, tgt_line, ref_markers[0], sys_markers[0], corrector)
    assert result[2] == CorrectionResult.CORRECTED
    correct_sys_line = "Úlla Árdalur has the least work experience of the three, who came to work on RÚV in 2019."
    assert result[0] == correct_sys_line
    correct_sys_tags = read_ner_tags(["P:0:12 O:77:80"])
    correct_sys_markers = to_ner_markers(correct_sys_tags, [correct_sys_line])
    assert result[1] == correct_sys_markers[0]
    result = embed_ner_tags(correct_sys_line, correct_sys_tags[0])
    assert result == "<P>Úlla Árdalur</P> has the least work experience of the three, who came to work on <O>RÚV</O> in 2019."

from islenska import Bin

from mt_named_entity.correct import CorrectionResult, Corrector


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

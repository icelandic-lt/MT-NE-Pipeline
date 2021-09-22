from islenska import Bin

from mt_named_entity.correct import Corrector


def test_variant_lookup():
    b = Bin()
    m = b.lookup_variants("Einari", "no", ("NF"))
    assert m[0].bmynd == "Einar"

def test_corrector_multiple_tokens():
    corrector = Corrector(should_correct_to_nomintaive_case=True)
    result = corrector.correct_icelandic_to_nominative_case("Einars Jónssonar")
    assert result == "Einar Jónsson"

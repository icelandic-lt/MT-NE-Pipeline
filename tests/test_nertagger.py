import pytest

from greynirseq.ner import nertagger


@pytest.mark.slow
def test_en_nertagger():
    test = "The insurgencies were effectively part of a violent class struggle initiated by the educated but disenchanted Sinhalese youth of the country, who were unable to prosper under post-Independence Governments. The age-old Sinhalese-Tamil ethnic animosities and tensions were exacerbated by the Tamil demand for autonomy and a separate state, and resulted in thirty years of civil war."
    result = list(nertagger.english_ner([test], device="cpu"))[0]

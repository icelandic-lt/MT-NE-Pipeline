from islenska import Bin


def test_variant_lookup():
    b = Bin()
    m = b.lookup_variants("Einari", "no", ("NF"))
    assert m[0].bmynd == "Einar"

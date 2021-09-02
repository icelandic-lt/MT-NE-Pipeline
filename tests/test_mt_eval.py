from greynirseq.ner.mt_eval import read_embedded_markers
from greynirseq.ner.ner_extracter import NERMarker


def test_read_embedded_markers(embedded_ner_tagged_sentences_is):
    actual_sents_markers, bad_markers = read_embedded_markers(
        embedded_ner_tagged_sentences_is, contains_model_marker=False
    )
    assert not bad_markers, "There are no bad markers."
    expected_sents_makers = [
        [NERMarker(0, 1, "P", "Guðrún"), NERMarker(5, 7, "P", "Einar Jónssonar")],
        [
            NERMarker(0, 1, "P", "Anna"),
            NERMarker(4, 5, "P", "Alexei"),
            NERMarker(5, 6, "P", "Pétri"),
            NERMarker(7, 8, "P", "Páli"),
        ],
    ]
    all(
        [
            expected == actual
            for sent_expected, sent_actual in zip(expected_sents_makers, actual_sents_markers)
            for expected, actual in zip(sent_expected, sent_actual)
        ]
    )


def test_specific_embedded_markers():
    test = "The insurgencies were effectively part of a violent class struggle initiated by the educated but disenchanted <M>Sinhalese</M> youth of the country, who were unable to prosper under <M>post</M>-<M>Independence</M> Governments. The age-old <M>Sinhalese</M>-<M>Tamil</M> ethnic animosities and tensions were exacerbated by the <M>Tamil</M> demand for autonomy and a separate state, and resulted in thirty years of civil war."
    markers, bad_markers = read_embedded_markers([test])
    assert not bad_markers


# ['59:<M>', '59:</M>', '59:<M>', '59:</M>', '143:<P>', '143:</P>', '266:<M>', '266:</M>', '392:<M>', '392:</M>', '773:<L>', '773:</L>', '808:<M>', '808:</M>', '860:<M>', '860:</M>', '962:<M>', '962:</M>', '1031:<M>', '1031:</M>', '1100:<L>', '1100:</L>', '1100:<L>', '1100:</L>', '1248:<M>', '1248:</M>', '1286:<O>', '1286:</O>', '1351:<M>', '1351:</M>', '1377:<M>', '1377:</M>', '1479:<M>', '1479:</M>', '1611:<M>', '1611:</M>', '1618:<M>', '1618:</M>', '1625:<M>', '1625:</M>', '1637:<L>', '1637:</L>', '1659:<O>', '1659:</O>', '1694:<L>', '1694:</L>', '1825:<M>', '1825:</M>', '1833:<M>', '1833:</M>']

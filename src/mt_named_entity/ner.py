import argparse
import sys
from dataclasses import dataclass
from typing import Generator, Iterable, List, Tuple

import spacy
import torch
import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
from spacy.gold import biluo_tags_from_offsets
from tokenizer import correct_spaces
from transformers import AutoModelForTokenClassification, AutoTokenizer

NER_RESULTS = Generator[Tuple[List[str], List[str], str], None, None]


@dataclass(frozen=True)
class NERTag:
    """A NER tag with character offsets."""

    tag: str
    start_idx: int
    end_idx: int


class EN_NER:
    def __init__(self, device):
        self.model = SequenceTagger.load("flair/ner-english-large")
        self.model.to(device)
        self.model.eval()

    def __call__(self, batch: Iterable[str]) -> List[List[NERTag]]:
        sentences = [Sentence(sent) for sent in batch]
        self.model.predict(sentences, mini_batch_size=32)
        # iterate over entities and print
        sentences_dict = map(lambda s: s.to_dict(tag_type="ner"), sentences)
        # {'text': 'George Washington went to Washington.',
        # 'entities': [
        #     {'text': 'George Washington', 'start_pos': 0, 'end_pos': 17, 'type': 'PER', 'confidence': 0.999},
        #     {'text': 'Washington', 'start_pos': 26, 'end_pos': 36, 'type': 'LOC', 'confidence': 0.998}
        # ]}

        return [
            [NERTag(entity["type"], entity["start_pos"], entity["end_pos"]) for entity in sent["entities"]]
            for sent in sentences_dict
        ]


def english_ner(lines_in: Iterable[str], device: str) -> List[List[NERTag]]:
    """NER tags a given collection sentences.

    Args:
        lines_in: The sentences should be given as a string, tokenized and joined by ' ' as the model expects.

    Returns:
        An iterable of a list of tokens, labels and a string representing the model used to NER tagging.
    """

    nlp = spacy.load("en_core_web_lg")

    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english").to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    label_list = [
        "O",  # Outside of a named entity
        "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        "I-MISC",  # Miscellaneous entity
        "B-PER",  # Beginning of a person's name right after another person's name
        "I-PER",  # Person's name
        "B-ORG",  # Beginning of an organisation right after another organisation
        "I-ORG",  # Organisation
        "B-LOC",  # Beginning of a location right after another location
        "I-LOC",  # Location
    ]

    def spacy_tok_ner(sent: str):
        doc = nlp(sent)
        j = doc.to_json()

        ranges = [(a["start"], a["end"]) for a in j["tokens"]]
        ents = j["ents"]

        tokens = []
        for range in ranges:
            tokens.append(sent[range[0] : range[1]])  # noqa

        entlocs = [(a["start"], a["end"], a["label"]) for a in ents]
        labels = biluo_tags_from_offsets(doc, entlocs)

        return tokens, labels

    def hugface_tok_ner(sequence: str) -> List[NERTag]:
        # Bit of a hack to get the tokens with the special tokens
        ner_tags = []
        encoded = tokenizer(text=sequence, return_offsets_mapping=True, return_tensors="pt")
        outputs = model(encoded.input_ids.to(device))[0]  # [0] refers to the NER head
        prediction_idxs = torch.argmax(outputs, dim=2).tolist()[0]  # Only get the first element in batch
        predicted_labels = [label_list[prediction_idx] for prediction_idx in prediction_idxs]
        offsets = encoded["offset_mapping"][0]
        for label, offset in zip(predicted_labels, offsets):
            if label != label_list[0]:  # we ignore "O"
                ner_tags.append(NERTag(label, *offset))
        return ner_tags

    for line in lines_in:
        using = "hf"
        if len(line.split()) < 400:
            tokens, ents = hugface_tok_ner(line)
        else:
            using = "sp"
            tokens, ents = spacy_tok_ner(line)
        assert len(ents) == len(tokens), "We expect the tokens to be of equal length to the labels"
        yield tokens, ents, using


def english_tok(lines_in: Iterable[str]) -> Iterable[str]:
    """Tokenizes English sentences. Can be split on spaces; " " to retrieve tokens."""
    from spacy.lang.en import English

    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.tokenizer
    for idx, line in enumerate(lines_in):
        line = line.strip()
        if not line:
            log.warning(f"Found empty line at index={idx}")
            continue
        yield " ".join(tok.text for tok in tokenizer(line))


def ner(lang: str, lines_iter: Iterable[str], device: str) -> NER_RESULTS:
    """Apply NER tagging on a collection of lines. Assumes input is tokenized."""
    if lang == "is":
        return icelandic_ner(lines_iter, device=device)
    elif lang == "en":
        return english_ner(lines_iter, device=device)
    else:
        raise ValueError(f"Unsupported language={lang}")


def tok(lang: str, lines_iter: Iterable[str]) -> Iterable[str]:
    """Tokenize a collection of lines."""
    if lang == "is":
        return icelandic_tok(lines_iter)
    elif lang == "en":
        return english_tok(lines_iter)
    else:
        raise ValueError(f"Unsupported language={lang}")


def detok(lines_iter: Iterable[str]) -> Iterable[str]:
    return (correct_spaces(line) for line in lines_iter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["is", "en"])
    parser.add_argument("--input", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("--output", nargs="?", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    f_in = args.input
    f_out = args.output

    log.info(f"NER tagging {args.input}->{args.output}")
    toks = tok(lang=args.language, lines_iter=f_in)
    tagged_iter = ner(lang=args.language, lines_iter=tqdm.tqdm(toks), device=args.device)
    for tokens, labels, using in tagged_iter:
        f_out.write(f"{' '.join(tokens)}\t{' '.join(labels)}\t{using}\n")

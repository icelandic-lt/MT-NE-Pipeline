import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Generator, Iterable, List, Tuple

import flair
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from greynirseq.cli.greynirseq import NER
from tokenizer import tokenize
from tokenizer.tokenizer import split_into_sentences

NER_RESULTS = Generator[Tuple[List[str], List[str], str], None, None]
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NERTag:
    """A NER tag with character offsets."""

    tag: str
    start_idx: int
    end_idx: int

    def __repr__(self) -> str:
        return f"{self.tag}:{self.start_idx}:{self.end_idx}"

    @staticmethod
    def from_str(a_str: str) -> "NERTag":
        """Parse a string representation of a NERTag."""
        tag, start_idx, end_idx = a_str.split(":")
        return NERTag(tag, int(start_idx), int(end_idx))


@dataclass(frozen=True)
class NERMarker(NERTag):
    """Hold a NER marker"""

    named_entity: str

    def __str__(self) -> str:
        return f"{self.tag}:{self.start_idx}:{self.end_idx}:{self.named_entity}"

    @staticmethod
    def from_str(a_str: str) -> "NERMarker":
        """Parse a string representation of a NERMarker."""
        tag, start_idx, end_idx, entity = a_str.split(":")
        return NERMarker(tag, int(start_idx), int(end_idx), entity)

    @staticmethod
    def from_tag(tag: NERTag, line: str) -> "NERMarker":
        """Parse a string representation of a NERMarker."""
        return NERMarker(tag.tag, tag.start_idx, tag.end_idx, line[tag.start_idx : tag.end_idx])


class EN_NER:
    def __init__(self, device, batch_size):
        flair.device = torch.device(device)
        self.model: SequenceTagger = SequenceTagger.load("flair/ner-english-large")  # type: ignore
        self.batch_size = batch_size

    def __call__(self, batch: Iterable[str]) -> List[List[NERTag]]:
        sentences = [Sentence(sent) for sent in batch]
        self.model.predict(sentences, mini_batch_size=self.batch_size)
        sentences_dict = list(map(lambda s: s.to_dict(tag_type="ner"), sentences))
        for sent in sentences_dict:
            for entity in sent["entities"]:
                if len(entity["labels"]) > 1:
                    log.error(f"Found two labels, be sure that they are in decreasing order:{entity['labels']}")
        return [
            [NERTag(entity["labels"][0].value, entity["start_pos"], entity["end_pos"]) for entity in sent["entities"]]
            for sent in sentences_dict
        ]


class IS_NER:
    def __init__(self, device, batch_size):
        self.model = NER(device, batch_size=batch_size, show_progress=True, max_input_words_split=100)

    def __call__(self, input) -> List[List[NERTag]]:
        all_lines = [line.strip() for line in input]
        all_tokens: List[List[str]] = []
        for line in all_lines:
            list_of_lines = list(split_into_sentences(line))
            tokens = []
            for a_line in list_of_lines:
                tokens.extend(a_line.split(" "))
            all_tokens.append(tokens)

        ner_tags = []
        tmp_tokens_file = f"tmp_tokens_{datetime.now()}"
        tmp_labels_file = f"tmp_labels_{datetime.now()}"
        with open(tmp_tokens_file, "w") as f_tokens:
            for tokens in all_tokens:
                f_tokens.write(" ".join(tokens) + "\n")
        with open(tmp_tokens_file, "r") as f_tokens, open(tmp_labels_file, "w") as f_labels:
            self.model.run(f_tokens, f_labels)
        with open(tmp_labels_file, "r") as f_labels:
            for line, labels, tokens in zip(all_lines, f_labels, all_tokens):
                label_list = [label for label in labels.strip().split(" ") if label != ""]
                ner_tags.append(self.remove_B(self.join_ner_tags(self.parse_ner_tags(line, tokens, label_list))))
        os.remove(tmp_labels_file)
        os.remove(tmp_tokens_file)
        return ner_tags

    @staticmethod
    def remove_B(ner_tags: List[NERTag]) -> List[NERTag]:
        """Remove B- tags from the beginning of the tag sequence."""
        fixed_ner_tags = []
        for ner_tag in ner_tags:
            if ner_tag.tag.startswith("B-"):
                fixed_ner_tags.append(NERTag(ner_tag.tag[2:], ner_tag.start_idx, ner_tag.end_idx))
        return fixed_ner_tags

    @staticmethod
    def join_ner_tags(ner_tags: List[NERTag]) -> List[NERTag]:
        """Join NER tags which start with I- to the B- tag infront, recursively.
        Assert that the tags to be joined have the same ending."""
        for idx, ner_tag in enumerate(ner_tags):
            if "I-" in ner_tag.tag:
                # It happens that the first tag is I-<tag>, we map it to B-<tag> and continue
                if idx == 0:
                    log.error(f"Found I- tag as a starting tag: {ner_tag.tag}")
                    log.error("Changing the I-tag to be a B-tag.")
                    ner_tag = NERTag("B-" + ner_tag.tag[2:], ner_tag.start_idx, ner_tag.end_idx)
                    ner_tags[idx] = ner_tag
                    return IS_NER.join_ner_tags(ner_tags)

                prev_ner_tag = ner_tags[idx - 1]
                # If the previous tag is B-<tag1> but we read I-<tag2> we map it to B-<tag1>I-<tag1> 
                if not prev_ner_tag.tag.endswith(ner_tag.tag[2:]):
                    log.error(f"Found I- tag with different ending than B- tag: {ner_tag.tag}, {prev_ner_tag.tag}")
                    log.error("Changing the I-tag to be consistent with the B-tag.")
                    ner_tag = NERTag("B-" + ner_tag.tag[2:], ner_tag.start_idx, ner_tag.end_idx)
                assert (
                    "B-" in prev_ner_tag.tag
                ), f"Found I- tag with no B- tag before it: {ner_tag.tag}, {prev_ner_tag.tag}"
                ner_tags[idx - 1] = NERTag(prev_ner_tag.tag, prev_ner_tag.start_idx, ner_tag.end_idx)
                ner_tags.pop(idx)
                return IS_NER.join_ner_tags(ner_tags)
        return ner_tags

    @staticmethod
    def parse_ner_tags(line: str, tokens: List[str], labels: List[str]) -> List[NERTag]:
        """Iterate through the labels and tokens, and for each token, we find the substring in the original line to get the indices.
        Return a list of NERTags which are not O."""
        extra_length_for_spaces = 3
        assert len(tokens) == len(
            labels
        ), f"Number of tokens and labels do not match: \ntokens={tokens}\nlabels={labels}\n"
        tags = []
        start_idx = 0
        additional_length = 0
        for token, label in zip(tokens, labels):
            found_idx = line.find(token, start_idx, start_idx + len(token) + extra_length_for_spaces)
            if found_idx == -1:
                # The tokenizer actually coalesces '% 44' to a single token, so we have to check for that.
                found_idx = line.find(token[:-1], start_idx, start_idx + len(token) + extra_length_for_spaces)
                additional_length = 1
                if found_idx == -1:
                    # The tokenizer actually coalesces '$ 10'  to a single token, so we have to check for that.
                    found_idx = line.find(token[1:], start_idx, start_idx + len(token) + extra_length_for_spaces)
                    additional_length = 1
                    if found_idx == -1:
                        raise ValueError(f"Could not find token: {token}, line: {line}")
                    else:
                        found_idx -= 2
            end_idx = found_idx + len(token) + additional_length
            if label != "O":
                assert found_idx < end_idx, f"Found end index before start index: {found_idx}, {end_idx}, {line}"
                tags.append(NERTag(label, found_idx, end_idx))
            start_idx = end_idx
            additional_length = 0
        return tags

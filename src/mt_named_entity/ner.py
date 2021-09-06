import logging
from dataclasses import dataclass
from typing import Generator, Iterable, List, Tuple

import flair
import torch
from flair.data import Sentence
from flair.models import SequenceTagger

NER_RESULTS = Generator[Tuple[List[str], List[str], str], None, None]
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NERTag:
    """A NER tag with character offsets."""

    tag: str
    start_idx: int
    end_idx: int


class EN_NER:
    def __init__(self, device):
        flair.device = torch.device(device)
        self.model: SequenceTagger = SequenceTagger.load("flair/ner-english-large") # type: ignore

    def __call__(self, batch: Iterable[str], batch_size) -> List[List[NERTag]]:
        sentences = [Sentence(sent) for sent in batch]
        self.model.predict(sentences, mini_batch_size=batch_size)
        sentences_dict = list(map(lambda s: s.to_dict(tag_type="ner"), sentences))
        for sent in sentences_dict:
            for entity in sent["entities"]:
                if len(entity["labels"]) > 1:
                    log.error(f"Found two labels, be sure that they are in decreasing order:{entity['labels']}")
        return [
            [NERTag(entity["labels"][0].value, entity["start_pos"], entity["end_pos"]) for entity in sent["entities"]]
            for sent in sentences_dict
        ]

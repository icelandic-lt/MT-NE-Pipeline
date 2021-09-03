import logging

import click
from tqdm import tqdm

from mt_named_entity import ner

log = logging.getLogger(__name__)


@click.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
@click.option("--lang", type=click.Choice(["en", "is"]), default="is")
def ner(inp, out, lang):
    """A command to NER tag input from stdin and write to stdout.
    Input has a sentence in each line.
    The output has a sentence in each line, with tokens separated with whitespace a tab char and NER tags separated with whitespace.
    Maintains empty lines."""
    log.info(f"NER tagging")
    logging.basicConfig(level=logging.INFO)
    toks = ner.tok(lang=lang, lines_iter=inp)
    tagged_iter = ner.ner(lang=lang, lines_iter=tqdm(toks), device="cuda")
    for tokens, labels, using in tagged_iter:
        f_out.write(f"{' '.join(tokens)}\t{' '.join(labels)}\t{using}\n")

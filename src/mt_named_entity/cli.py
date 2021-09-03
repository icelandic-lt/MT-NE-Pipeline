import logging

import click
from tqdm import tqdm

from mt_named_entity.ner import EN_NER

log = logging.getLogger(__name__)

@click.group()
def cli():
    pass

@cli.command()
@click.argument("inp", type=click.File("r"))
@click.argument("out", type=click.File("w"))
@click.option("--lang", type=click.Choice(["en", "is"]), default="is")
@click.option("--device", type=str, default="cpu")
@click.option("--idx_file", type=str, default=None)
def ner(inp, out, lang, device, idx_file):
    """A command to NER tag input from stdin and write to stdout.
    Input has a sentence in each line.
    The output has a sentence in each line, with tokens separated with whitespace a tab char and NER tags separated with whitespace.
    Maintains empty lines."""
    log.info(f"NER tagging")
    logging.basicConfig(level=logging.INFO)
    ner = EN_NER(device)
    inp = tqdm(inp)

    if idx_file:
        idx_file = open(idx_file, "w")
    for sent_ner_tag in ner(inp):
        out.write(" ".join([tag.tag for tag in sent_ner_tag]) + "\n")
        if idx_file:
            idx_file.write(" ".join([f"{tag.start_idx}:{tag.end_idx}" for tag in sent_ner_tag]) + "\n")
    if idx_file:
        idx_file.close()
    log.info(f"NER tagging done")

if __name__ == "__main__":
    cli()

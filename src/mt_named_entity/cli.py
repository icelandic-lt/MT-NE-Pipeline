import logging

import click
from tqdm import tqdm

from mt_named_entity.embed import embed_ner_tags, extract_ner_tags
from mt_named_entity.ner import EN_NER, NERTag

log = logging.getLogger(__name__)

@click.group()
def cli():
    pass

@cli.command()
@click.argument("inp", type=click.File("r"))
@click.argument("out", type=click.File("w"))
@click.option("--lang", type=click.Choice(["en", "is"]), default="is")
@click.option("--device", type=str, default="cpu")
@click.option("--batch_size", type=int, default=64)
@click.option("--idx_file", type=str, default=None)
def ner(inp, out, lang, device, batch_size, idx_file):
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
    for sent_ner_tag in ner(inp, batch_size=batch_size):
        out.write(" ".join([tag.tag for tag in sent_ner_tag]) + "\n")
        if idx_file:
            idx_file.write(" ".join([f"{tag.start_idx}:{tag.end_idx}" for tag in sent_ner_tag]) + "\n")
    if idx_file:
        idx_file.close()
    log.info(f"NER tagging done")


@cli.command()
@click.argument("original", type=click.File("r"))
@click.argument("ner_entities", type=click.File("r"))
@click.argument("ner_idxs", type=click.File("r"))
@click.argument("output", type=click.File("w"))
def embed(original, ner_entities, ner_idxs, output):
    """Embed the NER markers, based on the idxs, into the original text."""
    log.info(f"Embedding")
    logging.basicConfig(level=logging.INFO)
    ner_entities = tqdm(ner_entities)
    ner_idxs = tqdm(ner_idxs)
    original = tqdm(original)

    for sent_ner_tag, sent_ner_idx, sent_original in zip(ner_entities, ner_idxs, original):
        sent_ner_tags = sent_ner_tag.strip().split()
        sent_ner_idxs = sent_ner_idx.strip().split()
        sent_ner_idxs_tuples =  [tuple(map(int, idx.split(":"))) for idx in sent_ner_idxs] 
        sent = sent_original.strip()
        ner_tags = [NERTag(ner_tag, start_idx=start, end_idx=end) for ner_tag, (start, end) in zip(sent_ner_tags, sent_ner_idxs_tuples)]
        sent_embed = embed_ner_tags(sent, ner_tags)
        output.write(sent_embed + "\n")
    log.info(f"Embedding done")

@cli.command()
@click.argument("embedded_text", type=click.File("r"))
@click.argument("ner_entities", type=click.File("w"))
@click.argument("ner_idxs", type=click.File("w"))
@click.argument("clean_text", type=click.File("w"))
def extract_embeds(embedded_text, ner_entities, ner_idxs, clean_text):
    """Extract embedded entities and write them out."""
    log.info(f"Removing embeddings")
    logging.basicConfig(level=logging.INFO)
    embedded_text = tqdm(embedded_text)
    for line in embedded_text:
        try:
            clean_line, ner_tags = extract_ner_tags(line.strip())
        except ValueError as e:
            log.exception(e)
            clean_text.write(line)
            ner_entities.write("\n")
            ner_idxs.write("\n")
            continue
        clean_text.write(clean_line + "\n")
        ner_entities.write(" ".join([tag.tag for tag in ner_tags]) + "\n")
        ner_idxs.write(" ".join([f"{tag.start_idx}:{tag.end_idx}" for tag in ner_tags]) + "\n")

    log.info(f"Removal done")
if __name__ == "__main__":
    cli()

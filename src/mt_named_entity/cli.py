import logging
from collections import Counter
from pprint import pprint

import click
import sacrebleu
from tqdm import tqdm

import mt_named_entity.filter as filter
from mt_named_entity.embed import embed_ner_tags, extract_ner_tags
from mt_named_entity.ner import EN_NER, IS_NER, NERTag

log = logging.getLogger(__name__)


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument("inp", type=click.File("r"))
@click.argument("out", type=click.File("w"))
@click.option("--lang", type=click.Choice(["en", "is"]), default="is")
@click.option("--device", type=str, default="cpu")
@click.option("--batch_size", type=int, default=64)
def ner(inp, out, lang, device, batch_size):
    """A command to NER tag input file and write to output file.
    Input has a sentence in each line, not tokenized.
    The output has the string representation of all NERTags found in the corresponding sentence, separated by a space.
    The output maintains empty lines."""
    log.info(f"NER tagging")
    inp = tqdm(inp)
    if lang == "en":
        ner = EN_NER(device, batch_size)
    else:
        ner = IS_NER(device, batch_size)
    for sent_ner_tag in ner(inp):
        out.write(" ".join([str(tag) for tag in sent_ner_tag]) + "\n")
    log.info(f"NER tagging done")


@cli.command()
@click.argument("original", type=click.File("r"))
@click.argument("ner_entities", type=click.File("r"))
@click.argument("output", type=click.File("w"))
def embed(original, ner_entities, output):
    """Embed the NER markers into the original text."""
    log.info(f"Embedding")
    original = tqdm(original)

    for sent_ner_tag, sent_original in zip(ner_entities, original):
        sent_ner_tags_str = sent_ner_tag.strip().split()
        sent_ner_tags = [NERTag.from_str(tag) for tag in sent_ner_tags_str]
        sent = sent_original.strip()
        sent_embed = embed_ner_tags(sent, sent_ner_tags)
        output.write(sent_embed + "\n")
    log.info(f"Embedding done")


@cli.command()
@click.argument("embedded_text", type=click.File("r"))
@click.argument("ner_entities", type=click.File("w"))
@click.argument("clean_text", type=click.File("w"))
def extract_embeds(embedded_text, ner_entities, clean_text):
    """Extract embedded entities and write them out."""
    log.info(f"Removing embeddings")
    embedded_text = tqdm(embedded_text)
    for line in embedded_text:
        try:
            clean_line, ner_tags = extract_ner_tags(line.strip())
        except ValueError as e:
            log.exception(e)
            clean_text.write(line)
            ner_entities.write("\n")
            continue
        clean_text.write(clean_line + "\n")
        ner_entities.write(" ".join([str(tag) for tag in ner_tags]) + "\n")

    log.info(f"Removal done")


@cli.command()
@click.argument("src_text", type=click.File("r"))
@click.argument("tgt_text", type=click.File("r"))
@click.argument("src_entities", type=click.File("r"))
@click.argument("tgt_entities", type=click.File("r"))
@click.argument("src_text_out", type=click.File("w"))
@click.argument("tgt_text_out", type=click.File("w"))
@click.argument("src_entities_out", type=click.File("w"))
@click.argument("tgt_entities_out", type=click.File("w"))
def filter_text_by_ner(
    src_text, tgt_text, src_entities, tgt_entities, src_text_out, tgt_text_out, src_entities_out, tgt_entities_out
):
    """Filter the src and tgt based on the provided NER entities. Empty lines are not written out."""
    log.info(f"Filtering")
    src_text = tqdm(src_text)
    for sent_src_text, sent_tgt_text, sent_src_entities, sent_tgt_entities in zip(
        src_text, tgt_text, src_entities, tgt_entities
    ):
        sent_src_entities = [NERTag.from_str(a_str) for a_str in sent_src_entities.strip().split(" ") if a_str != ""]
        sent_tgt_entities = [NERTag.from_str(a_str) for a_str in sent_tgt_entities.strip().split(" ") if a_str != ""]
        if not sent_src_entities or not sent_tgt_entities:
            continue
        sent_entities = [sent_src_entities, sent_tgt_entities]
        # We map the named entities to a unified format, so that we can use the same filter function.
        sent_entities = [filter.map_named_entity_types(entities) for entities in sent_entities]
        # We filter out named entities we are not interested in.
        sent_entities = [filter.filter_named_entity_types(entities) for entities in sent_entities]
        # We then filter out sentences which do not have the same number of entity types.
        if not filter.filter_same_number_of_entity_types(*sent_entities):
            continue

        # The newline is still present.
        src_text_out.write(sent_src_text)
        tgt_text_out.write(sent_tgt_text)
        src_entities_out.write(" ".join([str(tag) for tag in sent_src_entities]) + "\n")
        tgt_entities_out.write(" ".join([str(tag) for tag in sent_tgt_entities]) + "\n")
    log.info(f"Filtering done")


@cli.command()
@click.argument("entities_file", type=click.File("r"))
def statistics(entities_file):
    """Get statistics about NER entities in a file."""
    log.info(f"Getting statistics")
    counter = Counter()
    for line in entities_file:
        entities = [NERTag.from_str(a_str) for a_str in line.strip().split(" ") if a_str != ""]
        counter.update([entity.tag for entity in entities])
    for key, value in sorted(counter.items()):
        click.echo(f"{key}\t{value}")

@cli.command()
@click.argument("entities_file", type=click.File("r"))
@click.argument("entities_file_normalized", type=click.File("w"))
def normalize(entities_file, entities_file_normalized):
    """Normalize the entity names."""
    log.info(f"Normalizing")
    for line in entities_file:
        entities = [NERTag.from_str(a_str) for a_str in line.strip().split(" ") if a_str != ""]
        normalized_entities = filter.map_named_entity_types(entities)
        entities_file_normalized.write(" ".join([str(tag) for tag in normalized_entities]) + "\n")


@cli.command()
@click.argument("ref_text", type=click.File("r"))
@click.argument("sys_text", type=click.File("r"))
@click.argument("ref_entities", type=click.File("r"))
@click.argument("sys_entities", type=click.File("r"))
def eval(ref_text, sys_text, ref_entities, sys_entities):
    log.info(f"BLEU score: {sacrebleu.corpus_bleu(sys_stream=sys_text, ref_streams=[ref_text])}")
    log.info(f"Ref NER markers: {get_markers_stats(ref_markers)}")
    log.info(f"Sys NER markers: {get_markers_stats(sys_markers)}")
    alignments = [align_markers(ref_marker, sys_marker) for ref_marker, sys_marker in zip(ref_markers, sys_markers)]
    if alignments:
        upper_bound_ner_alignments = min(
            sum(len(markers) for markers in ref_markers), sum(len(markers) for markers in sys_markers)
        )
        log.info("Metrics over all types:")
        for metric, value in get_metrics(alignments, upper_bound_ner_alignments).items():
            log.info(f"\t{metric}: {value:.3f}")
        groups = {marker.tag for markers in chain(ref_markers, sys_markers) for marker in markers}
        for group in groups:
            upper_bound_ner_alignments = min(
                sum(1 for markers in ref_markers for marker in markers if marker.tag == group),
                sum(1 for markers in sys_markers for marker in markers if marker.tag == group),
            )
            if upper_bound_ner_alignments:
                # Refs are marker_1
                group_alignments = [
                    [alignment for alignment in s_alignment if alignment.marker_1.tag == group]
                    for s_alignment in alignments
                ]
                log.info(f"Metrics over {group}:")
                for metric, value in get_metrics(group_alignments, upper_bound_ner_alignments).items():
                    log.info(f"\t{metric}: {value:.3f}")

    else:
        log.info("No alignments!")


if __name__ == "__main__":
    cli()

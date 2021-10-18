import logging
import re
from collections import Counter
from random import sample, shuffle
from typing import Dict, Iterable, List

import click
from tqdm import tqdm

from mt_named_entity.align import align_markers_by_jaro_winkler, align_markers_by_order
from mt_named_entity.correct import Corrector, correct_line

from .embed import embed_ner_entity, embed_ner_tags, extract_ner_tags
from .eval import ALIGNED, ALL_METRICS, DISTANCE, MATCHES, UPPER_BOUND, get_metrics
from .filter import ALL_TAGS, filter_named_entity_types, filter_same_number_of_entity_types, map_named_entity_types
from .ner import EN_NER, IS_NER, NERMarker, NERTag

log = logging.getLogger(__name__)

ALL_GROUPS = ["all"] + ALL_TAGS
METRIC_FIELDS = [f"{group}_{metric}" for group in ALL_GROUPS for metric in ALL_METRICS]


@click.group()
@click.option("--debug/--no_debug", default=False)
def cli(debug):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument("inp", type=click.File("r"))
@click.argument("out", type=click.File("w"))
@click.option("--tokens", type=int, default=250)
def shorten(inp, out, tokens):
    """Shorten lines in a file. If the number of tokens in the line exceeds spaces we throw it away."""
    log.info(f"Shortening lines in file")
    inp = tqdm(inp)
    for line in inp:
        if len(line.strip().split(" ")) > tokens:
            continue
        out.write(line)

@cli.command()
@click.argument("inp", type=click.File("r"))
@click.argument("out", type=click.File("w"))
def clean(inp, out):
    """Cleans lines in a file by removing multiple spaces and bad characters."""
    log.info(f"Cleaning file")
    inp = tqdm(inp)
    MULTIPLE_SPACES = re.compile(r" +")
    for line in inp:
        line = line.strip()
        line = line.replace("\xad", "")
        line = line.replace(u'\xa0', u' ')
        line = MULTIPLE_SPACES.sub(" ", line)
        out.write(line + "\n")


@cli.command()
@click.argument("inp_lang1", type=click.File("r"))
@click.argument("inp_lang2", type=click.File("r"))
@click.argument("out_lang1", type=click.File("w"))
@click.argument("out_lang2", type=click.File("w"))
@click.argument("total_num_lines", type=int)
@click.option("--fraction", type=float, default=0.05, help="Fraction of total lines in input to keep in sample.")
def sample_parallel(inp_lang1, inp_lang2, out_lang1, out_lang2, total_num_lines, fraction):
    log.info(f"Sampling parallel corpus.")
    num_lines_to_keep = int(total_num_lines * fraction)
    log.info(
        f"Total lines in input: {total_num_lines} with fraction: {fraction}. Resulting in: {num_lines_to_keep} lines."
    )
    inp_lang1 = tqdm(inp_lang1)
    indices = sorted(sample(range(total_num_lines), num_lines_to_keep))
    indices_index = 0
    for line_lang1, line_lang2, idx in zip(inp_lang1, inp_lang2, range(total_num_lines)):
        if idx == indices[indices_index]:
            out_lang1.write(line_lang1)
            out_lang2.write(line_lang2)
            indices_index += 1
            if indices_index == num_lines_to_keep:
                break


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
@click.argument("original", type=click.File("r"))
@click.argument("ner_entities", type=click.File("r"))
@click.argument("output", type=click.File("w"))
def unique_ner_entities(original, ner_entities, output):
    """Return all unique NER entities found in the original text."""
    log.info(f"Finding unique NER entities")
    original = tqdm(original)
    ner_entities = read_ner_tags(ner_entities)
    all_entities = set()
    for sent_ner_tags, sent_original in zip(ner_entities, original):
        sent = sent_original.strip()
        all_entities.update(embed_ner_entity(sent, ner_tag) for ner_tag in sent_ner_tags)

    for entity in sorted(all_entities):
        output.write(entity + "\n")
    log.info(f"Done")


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
    src_text_to_write = []
    tgt_text_to_write = []
    src_entities_to_write = []
    tgt_entities_to_write = []
    for sent_src_text, sent_tgt_text, sent_src_entities, sent_tgt_entities in zip(
        src_text, tgt_text, src_entities, tgt_entities
    ):
        sent_src_entities = [NERTag.from_str(a_str) for a_str in sent_src_entities.strip().split(" ") if a_str != ""]
        sent_tgt_entities = [NERTag.from_str(a_str) for a_str in sent_tgt_entities.strip().split(" ") if a_str != ""]
        if not sent_src_entities or not sent_tgt_entities:
            continue
        sent_entities = [sent_src_entities, sent_tgt_entities]
        # We map the named entities to a unified format, so that we can use the same filter function.
        sent_entities = [map_named_entity_types(entities) for entities in sent_entities]
        # We filter out named entities we are not interested in.
        sent_entities = [filter_named_entity_types(entities) for entities in sent_entities]
        if not sent_entities[0] or not sent_entities[1]:
            continue
        # We then filter out sentences which do not have the same number of entity types.
        src_entities, tgt_entities = filter_same_number_of_entity_types(*sent_entities)
        if not src_entities or not tgt_entities:
            continue
        # The newline is still present.
        src_text_to_write.append(sent_src_text)
        tgt_text_to_write.append(sent_tgt_text)
        src_entities_to_write.append(src_entities)
        tgt_entities_to_write.append(tgt_entities)

    assert len(src_text_to_write) == len(
        tgt_text_to_write
    ), f"The source text and tgt should have the same lengths. src={len(src_text_to_write)}, tgt={len(tgt_text_to_write)}"
    assert len(src_entities_to_write) == len(
        tgt_entities_to_write
    ), f"The source NEs and target should have the same lengths. src_ne={len(src_entities_to_write)}, tgt_ne={tgt_entities_to_write}"
    assert len(src_entities_to_write) == len(
        src_text_to_write
    ), f"The source text and NEs should have the same lengths. src_text={len(src_text_to_write)}, src_ne={len(src_entities_to_write)}"
    length = len(src_text_to_write)
    indices = list(range(length))
    shuffle(indices)
    for idx in indices:
        sent_src_text, sent_tgt_text, sent_src_entities, sent_tgt_entities = (
            src_text_to_write[idx],
            tgt_text_to_write[idx],
            src_entities_to_write[idx],
            tgt_entities_to_write[idx],
        )
        assert len(sent_src_entities) == len(
            sent_tgt_entities
        ), f"The source NEs and target NEs should have the same lengths. src_ne={len(sent_src_entities)}, tgt_ne={len(sent_tgt_entities)}"
        assert sent_src_text.strip() != ""
        assert sent_tgt_text.strip() != ""
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


def read_ner_tags(file_stream: Iterable[str]) -> List[List[NERTag]]:
    """Read the NER tags from a file."""
    return [[NERTag.from_str(a_str) for a_str in line.strip().split(" ") if a_str != ""] for line in file_stream]


@cli.command()
@click.argument("entities_file", type=click.File("r"))
@click.argument("entities_file_normalized", type=click.File("w"))
def normalize(entities_file, entities_file_normalized):
    """Normalize the entity names."""
    log.info(f"Normalizing")
    entities = read_ner_tags(entities_file)
    for sent_entities in entities:
        normalized_entities = map_named_entity_types(sent_entities)
        entities_file_normalized.write(" ".join([str(tag) for tag in normalized_entities]) + "\n")


@cli.command()
@click.argument("ref_text", type=click.File("r"))
@click.argument("sys_text", type=click.File("r"))
@click.argument("ref_entities", type=click.File("r"))
@click.argument("sys_entities", type=click.File("r"))
@click.option("--tsv/--no-tsv", default=False)
def eval(ref_text, sys_text, ref_entities, sys_entities, tsv):
    sys_text = [line.strip() for line in sys_text]
    ref_text = [line.strip() for line in ref_text]
    # log.info(f"BLEU score: {sacrebleu.corpus_bleu(sys_text, [ref_text])}")
    ref_entities = to_ner_markers(read_ner_tags(ref_entities), ref_text)
    sys_entities = to_ner_markers(read_ner_tags(sys_entities), sys_text)
    metrics: Dict[str, Dict[str, float]] = dict()
    alignments = [
        align_markers_by_jaro_winkler(ref_marker, sys_marker)
        for ref_marker, sys_marker in zip(ref_entities, sys_entities)
    ]
    if alignments:
        for group in ALL_GROUPS:
            # We count maximum alignments based on the ref
            upper_bound_ner_alignments = sum(
                1 for markers in ref_entities for marker in markers if marker.tag == group or group == "all"
            )
            # Refs are marker_1
            group_alignments = [
                [alignment for alignment in s_alignment if alignment.marker_1.tag == group or group == "all"]
                for s_alignment in alignments
            ]
            group_metrics = get_metrics(group_alignments, upper_bound_ner_alignments)
            metrics[group] = group_metrics

    else:
        raise ValueError("No alignments found")
    if tsv:
        click.echo(metric_values_to_tsv(metrics))
    else:
        log_metric_values(metrics)


def metric_values_to_tsv(metrics):
    a_str = ""
    for group in ALL_GROUPS:
        a_str += ",".join(f"{value:.3f}" for value in metrics[group].values()) + ","
    a_str += "\n"
    return a_str


def log_metric_values(metrics):
    for group in ALL_GROUPS:
        if metrics[group][ALIGNED] == 0.0:
            log.info(f"No alignments for {group}")
            print(f"No alignments for {group}")
        else:
            log.info(f"{group}")
            log.info(f"Alignment count: {metrics[group][ALIGNED]}")
            log.info(f"Alignment coverage: {metrics[group][ALIGNED]/metrics[group][UPPER_BOUND]}")
            log.info(f"Average alignment distance: {metrics[group][DISTANCE]/metrics[group][ALIGNED]}")
            log.info(f"Accuracy (exact match): {metrics[group][MATCHES]/metrics[group][ALIGNED]}")
            print(metrics[group][ALIGNED])
            print(metrics[group][ALIGNED] / metrics[group][UPPER_BOUND])
            print(metrics[group][DISTANCE] / metrics[group][ALIGNED])
            print(metrics[group][MATCHES] / metrics[group][ALIGNED])
            print()


@cli.command()
@click.argument("result_file", type=click.File("r"))
@click.option("--tsv/--no-tsv", default=False)
def combine_results(result_file, tsv):
    """Combine the results from a file accross groups. Write result to stdout."""
    group_metrics = {metric: 0 for metric in ALL_METRICS}
    metrics = {group: dict(group_metrics) for group in ALL_GROUPS}
    for line in result_file:
        values = [float(a) for a in line.strip().split(",") if a != ""]
        if len(values) == 0:
            continue
        idx = 0
        for group in ALL_GROUPS:
            for metric in ALL_METRICS:
                metrics[group][metric] += values[idx]
                idx += 1
    if tsv:
        click.echo(metric_values_to_tsv(metrics))
    else:
        log_metric_values(metrics)


def to_ner_markers(entities: List[List[NERTag]], text: List[str]) -> List[List[NERMarker]]:
    all_markers = []
    for idx, entities_line in enumerate(entities):
        all_markers.append([NERMarker.from_tag(tag, text[idx]) for tag in entities_line])
    return all_markers


@cli.command()
@click.argument("ref_text", type=click.File("r"))
@click.argument("sys_text", type=click.File("r"))
@click.argument("ref_entities", type=click.File("r"))
@click.argument("sys_entities", type=click.File("r"))
@click.option("--tag", type=str, default="P")
def show_examples(ref_text, sys_text, ref_entities, sys_entities, tag):
    """Show examples of alignments of a given tag type"""
    sys_text = [line.strip() for line in sys_text]
    ref_text = [line.strip() for line in ref_text]
    # Read the NER tags and map them to NERMarkers

    ref_entities = to_ner_markers(read_ner_tags(ref_entities), ref_text)
    sys_entities = to_ner_markers(read_ner_tags(sys_entities), sys_text)
    alignments = [
        align_markers_by_jaro_winkler(ref_marker, sys_marker)
        for ref_marker, sys_marker in zip(ref_entities, sys_entities)
    ]
    group_alignments = [
        [alignment for alignment in s_alignment if alignment.marker_1.tag == tag] for s_alignment in alignments
    ]
    assert len(group_alignments) == len(ref_text)
    for idx, s_alignment in enumerate(alignments):
        if len(s_alignment) > 0:
            if any(alignment.marker_1.tag == tag for alignment in s_alignment):
                for alignment in s_alignment:
                    print(alignment)
                    print(ref_text[idx])
                    print(sys_text[idx])


@cli.command()
@click.argument("ref_text", type=click.File("r"))
@click.argument("sys_text", type=click.File("r"))
@click.argument("ref_entities", type=click.File("r"))
@click.argument("sys_entities", type=click.File("r"))
@click.argument("sys_text_corrected", type=click.File("w"))
@click.option(
    "--to_nominative_case/--no_to_nominative_case",
    default=False,
    help="Attempt to convert named-entities to nominative case. Only works if NE is in BÍN. Works best if reference is Icelandic and system is English.",
)
@click.option(
    "--corrections_tsv",
    type=str,
    default=None,
    help="A filepath to a tsv with two columns containing corrections. First column is to match reference NE, second column is used to replace system NE.",
)
def correct(ref_text, sys_text, ref_entities, sys_entities, sys_text_corrected, to_nominative_case, corrections_tsv):
    """Correct the sys_text named entities according to options specified"""
    sys_text = [line.strip() for line in sys_text]
    ref_text = [line.strip() for line in ref_text]
    # Read the NER tags and map them to NERMarkers
    ref_markers = to_ner_markers(read_ner_tags(ref_entities), ref_text)
    sys_markers = to_ner_markers(read_ner_tags(sys_entities), sys_text)
    corrections = {}
    if corrections_tsv:
        corrections = read_corrections(corrections_tsv)
    correcter = Corrector(should_correct_to_nomintaive_case=to_nominative_case, corrections=corrections)
    corrected_sys_text = []
    for ref_line, sys_line, ref_marker, sys_marker in zip(ref_text, sys_text, ref_markers, sys_markers):
        corrected_sys_line = correct_line(ref_line, sys_line, ref_marker, sys_marker, correcter)
        corrected_sys_text.append(corrected_sys_line)
    sys_text_corrected.write("\n".join(corrected_sys_text))
    log.info("Correction statistics")
    log.info(correcter.correction_statistics)


def read_corrections(filepath: str) -> Dict[str, str]:
    corrections = {}
    with open(filepath, "r") as corrections_file:
        for line in corrections_file:
            line = line.strip()
            if line == "":
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid line in corrections file: {line}")
            corrections[parts[0]] = parts[1]
    return corrections


if __name__ == "__main__":
    cli()

# Named Entity Processing Pipeline for NMT

In training data for NMT (neural machine translation) systems it is of benefit to have a large and varried corpus. Unfortunately this is not often the case. This submodule implements a pipeline for tagging, filtering, matching, substituing/correcting and evaluating translation of named entities in a parallel English to Icelandic corpus.

## Installation
```
pip install git+https://github.com/mideind/MT-NE-Pipeline
```

## Name Tagging
For Icelandic NER the included IceBERT-NER model is used. For english we use Flair.

The following command accepts as input a txt file which has a sentence (or multiple) per line and writes out the NEs in the output file. The command will preserve empty lines.

```tests/testdata/example.is
Guðrún fór í heimsókn til Einars Jónssonar.
Anna fékk gjöf frá Alexei, Pétri og Páli.

Núna með Tómar Línur, takk Joe!

Ha?

```
Running the NER on the example file and writing output to `example.is.ner` (use `-` to specify stdout).
```bash
mt ner tests/data/example.is example.is.ner --lang is
cat example.is.ner
# Produces
Person:0:6 Person:26:42
Person:0:4 Person:19:25 Person:27:32 Person:36:40

Organization:9:20 Person:27:30


```
The NEs are written to `example.is.ner` in which each line corresponds to a line in the input. The NEs are formatted as `label:start_char_idx:end_char_idx`, i.e. the span of the label. The spans are separated with space. The BIO-markers have been joined together to create the span.

For more options (GPU/batch_size) call `mt ner --help`

Similarly, for English:
```tests/data/example.en
Guðrún visited Einars Jónssonar.
Anna got a gift from Pétri, Páli and Alexei.

Now with Empty Lines, thanks Joe!

Huh?
```

```bash
mt ner tests/data/example.en example.en.ner --lang en
cat example.en.ner
# Produces
PER:0:6 PER:15:31
PER:0:4 PER:21:26 PER:28:32 PER:37:43

MISC:9:20 PER:29:32


```
Notice that the taggers do not produce the same tag sets.

## Unifying tag sets
To be able to filter and/or align NE markers we need to unify the tag sets.
```
mt normalize example.en.ner example.en.ner-norm
mt normalize example.is.ner example.is.ner-norm
```

We can also embed the NEs directly into the sentences:
```
mt embed tests/data/example.is example.is.ner -
mt embed tests/data/example.en example.en.ner -
```

## Filtering based on NEs
Filtering is based on parallel data and works as follows
- Lines with no NEs are removed
- NE tag sets are normalized (like above)
- NE tags which are not Organization, Location or Person are filtered out
- Lines with unequal number of tags in each group are filtered out
- Then all the remaining lines are shuffled.

```bash
mt filter-text-by-ner tests/data/example.is tests/data/example.en example.is.ner example.en.ner example.is.filtered example.en.filtered example.is.ner.filtered example.en.ner.filtered
# Check the results
cat example.??*.filtered
Guðrún visited Einars Jónssonar.
Anna got a gift from Pétri, Páli and Alexei.
P:0:6 P:15:31
P:0:4 P:21:26 P:28:32 P:37:43
Guðrún fór í heimsókn til Einars Jónssonar.
Anna fékk gjöf frá Alexei, Pétri og Páli.
P:0:6 P:26:42
P:0:4 P:19:25 P:27:32 P:36:40
```
Only two lines remain.

## Correction/Substition
In our example, the English sentences is considered to be incorrect translations of the Icelandic sentences. They are incorrect because the names are not in nominative case. We will now correct this.
```
mt correct example.is.filtered example.en.filtered example.is.ner.filtered example.en.ner.filtered example.is.corrected --to_nominative_case
# example.is.corrected
Guðrún visited Einar Jónsson.
Anna got a gift from Alexei, Pétur and Páll.
```
Additionally, a dictionary can be provided to make manually corrections. The dictionary is used as a first correction resort.

```
mt correct example.is.filtered example.en.filtered example.is.ner.filtered example.en.ner.filtered example.is.corrected --to_nominative_case --corrections_tsv tests/data/corrections.tsv
# example.is.corrected
Guðrún visited Einar Jónsson.
Anna got a gift from Alexei Sergov, Pétur and Páll.
```
## MT evaluation

To evaluate an MT system w.r.t. BLEU run:
```
# This should give a perfect score.
lang=en
ref=testdata/example.ner-ext.$lang
sys=testdata/example.$lang
python mt_eval.py --ref $ref --ref-contains-entities --sys $sys --tgt_lang $lang
```
This will do the following:
- Read the NER markers from the REF.
- Report the BLEU score on the cleaned REF and SYS (as is).
- Run a NER on the SYS.
- Report on NER alignment: 
  - Alignment count: How many NEs we were able to match between REF and SYS.
  - Alignment coverage: The fraction of NEs which we were able to able to align, from 0-1, **1 is best**. If REF and SYS do not contain equal counts of NEs, we use the smaller count.
  - Average alignment distance: The average distance in the alignment, from 0-1, **0 is best**.
  - Accuracy: The fraction exact matches in the alignment (string comparison).
- Run the report on each distinct tag found both REF and SYS.

This evaluation can be run with any combination of --ref/sys-contains-entities.

## Analyzing and pairing

(This can be skipped) The next step aligns the two tagged files, and optionally prints some statistics. This step is run automatically by the filtering but can be ran on its own.

```bash
python aligner.py --is_ent testdata/is.ner --en_ent testdata/en.ner --output testdata/alignment.tsv
```

The columns are `ner_tagger_1, source_1, ner_tagger_2, source_2, match_code, max_distance (1-JarWink), alignment spans`

```
is		hf		1	0.06999999999999995	0:1:Person-5:6:PER 5:7:Person-0:2:PER
is		hf		1	0.12	0:1:Person-0:1:PER 4:5:Person-9:10:PER 6:7:Person-5:6:PER 8:9:Person-7:8:PER
```

## Filtering and POS tagging
This step parses the named files, aligns entities and pos tags them.

```bash
python postagger.py --is_ent testdata/is.ner --en_ent testdata/en.ner --output testdata/en_is.pos.tsv
```

The resulting file contains tags indicating which entity ID and part of speech (POS) a given name has in the Icelandic side.

```
<e:0:nkee-s:>Einar Jónsson</e0> was visited by <e:1:nven-s:>Guðrún</e1> .	<e:1:nven-s:>Guðrún</e1> fór í heimsókn til <e:0:nkee-s:>Einars Jónssonar</e0> .
<e:0:nven-s:>Anna</e0> got a gift from <e:1:nkeþ-s:>Pétur</e1> , <e:2:nkeþ-s:>Páll</e2> and <e:3:nkeþ-s:>Alexei</e3> .	<e:0:nven-s:>Anna</e0> fékk gjöf frá <e:3:nkeþ-s:>Alexei</e3> , <e:1:nkeþ-s:>Pétri</e1> og <e:2:nkeþ-s:>Páli</e2> .
```

## Substituting

Finally, given a list of tab separated genders (kk and kvk) and sufficient names such as 

```
kk  Þröstur Helagson
kk  Jón Jónsson
kk  Bubbi Morthens
kk  Ingvar Gunnarsson
kvk Sigga
kvk Sigríður Einarsdóttir
```

we can then generate a synthetic parallel corpus with randomly inserted names (full names and first names) using

```bash
python patcher.py --input testdata/en_is.pos.tsv --output testdata/en_is.synth.tsv --names testdata/names.txt
```

which outputs

```
Jón was visited by Sigríður .	Sigríður fór í heimsókn til Jóns .
Sigga got a gift from Bubbi Morthens , Ingvar and Jón .	Sigga fékk gjöf frá Jóni , Bubba Morthens og Ingvari .
```

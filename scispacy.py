#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some simple usage examples of the spaCy library

The original Colab notebook where this script came from:
    https://colab.research.google.com/drive/1O5qxkgvB3x80PuOo6EbVZnw55fnd_MZ3?usp=sharing

Colab notebook on stanza-bio:
    https://colab.research.google.com/drive/1AEdAzR4_-YNEClAB2TfSCYmWz7fIcHIO?usp=sharing

This script requires the 'stanza' and 'nltk' modules, I'd highly recommend you
install both libraries under an isolated python virtual environment and avoid
borking your system's python installation

Some extra info:
    Python version: 3.8.6
    OS: arch linux
"""
# Importing modules
import os
from pathlib import Path
import re
import shutil
from typing import List

import spacy
from spacy import displacy
from spacy.matcher import Matcher
from spacy.util import filter_spans


# Initializing the document annotator
nlp = spacy.load('en_core_sci_sm')

# Defining the text
# The text below was extracted from the 2019 BioNLP OST `BB-Rel` task, document
# `BB-rel-14633026.txt` from the Development dataset.
#
# Task URL: https://sites.google.com/view/bb-2019/dataset#h.p_n7YHdPTzsDaj
text = """
Characterization of a mosquitocidal Bacillus thuringiensis serovar sotto strain
isolated from Okinawa, Japan. To characterize the mosquitocidal activity of
parasporal inclusions of the Bacillus thuringiensis serovar sotto strain
96-OK-85-24, for comparison with two well-characterized mosquitocidal strains.
The strain 96-OK-85-24 significantly differed from the existing mosquitocidal
B. thuringiensis strains in: (1) lacking the larvicidal activity against Culex
pipiens molestus and haemolytic activity, and (2) SDS-PAGE profiles,
immunological properties and N-terminal amino acid sequences of parasporal
inclusion proteins. It is clear from the results that the strain 96-OK-85-24
synthesizes a novel mosquitocidal Cry protein with a unique toxicity spectrum.
This is the first report of the occurrence of a mosquitocidal B. thuringiensis
strain with an unusual toxicity spectrum, lacking the activity against the
culicine mosquito.
"""

print(text)

# Removing newlines
# The line below will replace newlines with a single whitespace, then any
# trailing spaces will be trimmed. We'll leave sentence segmentation for the
# trained model to handle.
text = re.sub(r'\n+', ' ', text).strip()
print(text)

# Annotating the document
# Just call `nlp(STRING)` and that's pretty much it
doc = nlp(text)

# Tokenization, Lemmas and Part-of-Speech (PoS) and Sentence Segmentation
# After annotating the text, the resulting `doc` object can be iterated for its
# collection of tokens and their attributes.
#
# Alternatively, the tokens can be accessed from each of the docs sentences in
# `doc.sents`
#
# https://spacy.io/api/token#attributes
print('  ID TEXT                 LEMMA                  UPOS POS    STOPWORD')
for (i, token) in enumerate(doc):
    head = 'ROOT' if token.head.i == token.i else token.head.text

    print(f'{i+1:>4} {token.text:<20} {token.lemma_:<20} {token.pos_:>6} ' +
          f'{token.tag_:<6} {str(token.is_stop):<10}')

for (i, sent) in enumerate(doc.sents):
    print(f'SENTENCE {i+1}: {sent.text}')
    print([t.text for t in sent], end='\n\n')

"""# Semantic Dependency Parsing
Semantic dependency information can be accessed at the token level
"""

print('  ID                 TEXT     DEP     HEAD TEXT        HEAD ID')
for (i, token) in enumerate(doc):
    head = 'ROOT' if token.head.i == token.i else token.head.text
    print(f'{i+1:>4} {token.text:>20} <{token.dep_:-^10} {head:<20}',
          f'{token.head.i:>3}')

# Saving semantic dependencies to SVG files
# Deleting previously saved files, if any
if os.path.exists('spacy-deps'):
    shutil.rmtree('spacy-deps')

os.mkdir('spacy-deps')
for (i, sent) in enumerate(doc.sents):
    svg_path = os.path.join('spacy-deps', f'sentence-{i+1}.svg')
    svg = displacy.render(sent, style='dep', jupyter=False)
    output_path = Path(svg_path)
    output_path.open('w', encoding='utf-8').write(svg)

# Noun Chunks
# Below, for each noun chunk in the document we display
#
# * The chunk's raw text;
# * The token in the chunk closest to the sentence's root;
# * The chunk's root token dependency to its head;
# * The text from the head of the chunk's root;
print('    NOUN CHUNK TEXT                                ROOT TEXT',
      'HEAD DEP  HEAD TEXT')
for (i, chunk) in enumerate(doc.noun_chunks):
    if chunk.root.head.i == chunk.root.i:
        # If the chunk's root token is also the sentence's root
        chunk_root_head_text = 'ROOT'
    else:
        chunk_root_head_text = chunk.root.head.text

    print(f'{i+1:>3} {chunk.text:<35} {chunk.root.text:>20}',
          f'<{chunk.root.dep_:-^10} {chunk_root_head_text}')


# Named Entity Recognition (NER)
# Using the available NER models on the same text data
#
# * CRAFT;
# * BC5CDR;
# * JNLPBA;
# * BIONLP13CG.
def show_ner(text: str, ner_model: str):
    """
    Just a shortcut to annotate the text with the given NER model
    """
    nlp = spacy.load(ner_model)
    doc = nlp(text)

    print(f'\nNER MODEL: {ner_model}')
    print('                TYPE TEXT')
    for ent in doc.ents:
        print(f'{ent.label_:>20} {ent.text}')


# CRAFT corpus
show_ner(text, 'en_ner_craft_md')

# BC5CDR corpus
show_ner(text, 'en_ner_bc5cdr_md')

# JNLPBA corpus
show_ner(text, 'en_ner_jnlpba_md')

# BIONLP13CG corpus
show_ner(text, 'en_ner_bionlp13cg_md')

# Rule-Based Matching
# Using the `<DT>?<JJ.*>*<NN.*>+` POS tag pattern example from the nltk book to
# capture noun chunks, where:
#
#  * `?` means zero or one of the previous pattern;
#  * `*` means zero or more of the previous pattern;
#  * `+` means one or more of the previous pattern;
#  * `.` means any (single) character.
#  * `<DT>?` - An optional determiner;
#  * `<JJ.*>*` - Zero or more adjectives;
#  * `<NN.*>+` - One or more nouns.
#
#  nltk-book: https://www.nltk.org/book/ch07.html

# Loading the model
# Using a model without NER capabilities to annotate the text
nlp = spacy.load('en_core_sci_sm')
doc = nlp(text)

# Displaying the generic entities
# The main pipelines `en_core_sci_sm|md|lg` come with the generic `ENTITY`
# label
Document = spacy.tokens.Doc
Span = spacy.tokens.Span


def print_all_entities(doc: Document):
    """
    Just a simple shortcut to list a document's entities

    Arguments:
        doc: Document
            An annotated spacy document
    """
    print(' ID LABEL      TEXT')
    for (i, ent) in enumerate(doc.ents):
        print(f'{i+1:>3} {ent.label_:<10} {ent.text:<35}')


print_all_entities(doc)


# Matching chunks to POS tag rules
def get_matched_pos_chunks(doc, pattern):
    """
    Get the list of chunks from the document that match the pattern

    Overlapping spans will be filtered and the longest ones will be
    returned on the list, for example, the text span

        a mosquitocidal Bacillus thuringiensis

    with PoS Tags

        DT JJ JJ NN

    will yield the following matches:

        DT JJ JJ NN - a mosquitocidal Bacillus thuringiensis
           JJ JJ NN -   mosquitocidal Bacillus thuringiensis
              JJ NN -                 Bacillus thuringiensis
                 NN -                          thuringiensis

    This function will ignore the shorter overlaps and return the
    longer. Not ideal but it gets the job done on this specific
    case.

    Arguments:
        doc: Document
            An annotated spacy document
        pattern: Dict
            A matcher pattern dictionary

    Returns: List[Span]
        The list matching chunks
    """
    matcher = Matcher(nlp.vocab)
    matcher.add("CHUNKS", None, pattern)
    matches = matcher(doc)
    chunk_spans = list()

    for (i, (match_id, start, end)) in enumerate(matches):
        span = doc[start:end]
        chunk_spans.append(span)

    longest_spans = filter_spans(chunk_spans)
    return longest_spans


def pretty_print_chunks(chunk_spans: List[Span]):
    """
    Just a quick way to do a formatted print on a list of spacy Spans

    Arguments:
        chunk_spans: List[Span]
            A list of chunk spans
    """
    print('      CHUNK POS TAGS | CHUNK TEXT')
    for (i, span) in enumerate(chunk_spans):
        pos_tags = ' '.join(t.tag_ for t in span)
        print(f'{i+1:>4} {pos_tags:>15} | {span.text}')


# The pattern definitions and results
# Creating the dictionary containing the `<DT>?<JJ.*>*<NN.*>+` pattern and
# filtering the annotated document for matching chunks.
pattern = [
    {"TAG": "DT", "OP": "?"},               # <DT>?
    {"TAG": {"REGEX": "JJ.*"}, "OP": "*"},  # <JJ.*>*
    {"TAG": {"REGEX": "NN.*"}, "OP": "+"}   # <NN.*>+
]

chunk_spans = get_matched_pos_chunks(doc, pattern)
pretty_print_chunks(chunk_spans)

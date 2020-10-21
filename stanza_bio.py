#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some simple usage examples of the stanfordnlp Stanza library

The original Colab notebook where this script came from:
    https://colab.research.google.com/drive/1AEdAzR4_-YNEClAB2TfSCYmWz7fIcHIO?usp=sharing

Colab notebook on scispaCy:
    https://colab.research.google.com/drive/1O5qxkgvB3x80PuOo6EbVZnw55fnd_MZ3?usp=sharing

This script requires the 'stanza' and 'nltk' modules, I'd highly recommend you
install both libraries under an isolated python virtual environment and avoid
borking your system's python installation

Some extra info:
    Python version: 3.8.6
    OS: arch linux
"""

# Importing modules
import re
import stanza
import nltk

# The path where downloads should be saved. By default it points to your user's
# HOME directory on linux/macos, no idea where it does it on windows.
STANZA_DOWNLOAD_DIR = './stanza_resources'

# Downloading the biomedical models
# Our main pipeline, trained with the CRAFT corpus
stanza.download('en', dir=STANZA_DOWNLOAD_DIR, package='craft')

# The NER models
stanza.download(lang='en', dir=STANZA_DOWNLOAD_DIR, package='jnlpba')
stanza.download(lang='en', dir=STANZA_DOWNLOAD_DIR, package='linnaeus')
stanza.download(lang='en', dir=STANZA_DOWNLOAD_DIR, package='s800')

# Initializing the document annotator
nlp = stanza.Pipeline(lang='en', dir=STANZA_DOWNLOAD_DIR, package='craft')

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
# An annotated document will have the following [structure][stanza-objs]:
# * A `document` contains `sentences`;
# * A `sentence` contains `tokens`/`words`;
# * A `token` contains one of more `words`;

# note: On stanza there is a distiction between a [Token][stanza-token] and a
# [Word][stanza-word] object.

# Unlike `spacy`, in order to access a word's properties (e.g.: lemma, PoS
# tags, etc.) you must iterate over the document's sentences and then iterate
# over each sentences to get their tokens/words (wich also means their token
# IDs are relative to the sentences they're from, you'll see down below). In
# general, a stanza code looks something like this:

# for sent in doc.sentences:
#     # Operating on the document sentences
#     # At this level you get the semantic dependencies
#
#     for token in sent.tokens:
#         # Operating on the sentence's tokens
#
#     for word in sent.words:
#         # Operating on the sentence's words
#
# https://stanfordnlp.github.io/stanza/data_objects.html
# https://stanfordnlp.github.io/stanza/data_objects.html#token
# https://stanfordnlp.github.io/stanza/data_objects.html#word
for (i, sent) in enumerate(doc.sentences):
    print(f'SENTENCE {i+1}')
    print('  ID                 TEXT LEMMA                 UPOS POS')
    for word in sent.words:
        print(f'{word.id:>4} {word.text:>20} {word.lemma:<20} {word.upos:>5}',
              f'{word.xpos:<5}')
    print()

# Semantic Dependency Parsing
# Semantic dependency information can be accessed at sentence level
print('  ID                 TEXT     DEP     HEAD TEXT        HEAD ID')
for (i, sent) in enumerate(doc.sentences):
    print(f'SENTENCE {i+1}')
    print('  ID            WORD TEXT <---DEP--- HEAD TEXT              ID')
    for dep in sent.dependencies:
        # Using 'Source' and 'Target' here as a reference to the semantic
        # dependency arrow direction
        [src_word, deprel, tgt_word] = dep
        print(f'{tgt_word.id:>4} {tgt_word.text:>20} <{deprel:-^9}',
              f'{src_word.text:<20} {src_word.id:>4}')
    print()

# Noun Chunks
# The stanza framework has no built-in chunking, instead we'll be using the
# `nltk` module and its example noun chunk grammar:
#
#   <DT>?<JJ.*>*<NN.*>+
#
#  * `<DT>?` - An optional determiner;
#  * `<JJ.*>*` - Zero or more adjectives;
#  * `<NN.*>+` - One or more nouns.
#
#  where:
#  * `?` means zero or one of the previous pattern;
#  * `*` means zero or more of the previous pattern;
#  * `+` means one or more of the previous pattern;
#  * `.` means any (single) character.
#
# https://www.nltk.org/book/ch07.html


def print_chunks(doc: stanza.Document, grammar: str,
                 print_full_tree: bool = True):
    """
    Print a document's chunks

    Arguments:
        doc: stanza.Document
            An (PoS) annotated document

        grammar: str
            An nltk chunk grammar regular expression

        print_full_tree: True|False
            If true, print the whole tree, else print only the matching grammar
            chunks
    """
    cp = nltk.RegexpParser(grammar)

    for (i, sent) in enumerate(doc.sentences):
        print(f'SENTENCE {i+1}')
        sentence = [(w.text, w.xpos) for w in sent.words]
        chunk_tree = cp.parse(sentence)

        if print_full_tree is True:
            print(chunk_tree, end='\n\n')
        else:
            for subtree in chunk_tree.subtrees():
                if subtree.label() == 'NP':
                    print(subtree)
            print()


grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}'
print_chunks(doc, grammar, print_full_tree=False)
print_chunks(doc, grammar, print_full_tree=True)

# Named Entity Recognition (NER)
# From stanza's available NER models we'll test the JNLPBA, Linnaeus and S800
# models
#
# https://stanfordnlp.github.io/stanza/available_biomed_models.html#biomedical--clinical-ner-models


def show_ner(text: str, ner_model: str):
    """
    Just a shortcut to annotate the text with the given NER model
    """
    nlp = stanza.Pipeline(lang='en', package='craft',
                          dir=STANZA_DOWNLOAD_DIR,
                          processors={'ner': ner_model})
    doc = nlp(text)

    print(f'\nNER MODEL: {ner_model}')
    print('TYPE       TEXT')
    for ent in doc.entities:
        print(f'{ent.type:<10} {ent.text}')


show_ner(text, 'jnlpba')
show_ner(text, 'linnaeus')
show_ner(text, 's800')

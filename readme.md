# Biomedical Parsers
Here are some example scripts that hopefully can introduce you to NLP of biomedical texts with [scispaCy][scispacy-home] and [stanza-bio][stanza_bio-home].

Preferably, you should try running the code from the provided colab notebooks since they require the least ammount of effort to setup and run but you can try running the scripts on your machine after you install their requirements

# Colab notebooks
The scripts were originally written on the colab notebooks below

stanza-bio: https://colab.research.google.com/drive/1AEdAzR4_-YNEClAB2TfSCYmWz7fIcHIO?usp=sharing

scispaCy: https://colab.research.google.com/drive/1O5qxkgvB3x80PuOo6EbVZnw55fnd_MZ3?usp=sharing

# Requirements to run the scripts locally
*   Both scripts require the python version 3.
*   The `stanza-bio.py` script requires the `stanza` and `nltk` modules.
*   The `scispacy.py` script requires the `scispacy` module and the `jupyter-notebook` client.

You'll have to find out how to install the requirements on your machine, preferably under a [virtual environment][venv]. Note that the scripts were tested under python version `3.8.6` on arch linux.

# scispaCy requirements installation
The scispaCy script requires the scispacy module and its dependencies, as well as the scientific models:

```zsh
# Install the scispacy and its dependencies
pip install scispacy

# Download the core pipeline
pip install 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz'

# Download the NER models
pip install 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_craft_md-0.3.0.tar.gz'
pip install 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_jnlpba_md-0.3.0.tar.gz'
pip install 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_bc5cdr_md-0.3.0.tar.gz'
pip install 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_bionlp13cg_md-0.3.0.tar.gz'
```

Also, you'll need to install the `jupyter-notebook` matching your operating system, incase you want to use the graph visualization tools.

# stanza-bio requirements installation
The stanza script only requires the `stanza` and `nltk` modules:

```zsh
pip install stanza nltk
```

[scispacy-home]: https://allenai.github.io/scispacy/
[stanza_bio-home]: https://stanfordnlp.github.io/stanza/
[venv]: https://towardsdatascience.com/virtual-environments-104c62d48c54

# Data

WikiText-103 ([Merity et al., 2017](https://openreview.net/forum?id=Byj72udxe))
is loaded as `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`.
AG News ([Zhang et al., 2015](https://papers.nips.cc/paper_files/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html))
is loaded as `fancyzhx/ag_news`. Both are downloaded through the `datasets`
package on the first run. The experiments use the `text` field and never use
AG News category labels.

The frozen
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
encoder produces 384-dimensional vectors, normalized before storage as
float16 tensors. The encoder is used for preprocessing and is not trained
with the associative-memory model. The relevant methods are Sentence-BERT
([Reimers and Gurevych, 2019](https://aclanthology.org/D19-1410/)) and MiniLM
([Wang et al., 2020](https://papers.nips.cc/paper_files/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)).
BibTeX entries are in [references.bib](../references.bib).

Each original text is a key. Its query is formed by independently dropping
words with probability 0.1, subject to the minimum query length defined in
the experiment. A SHA-256 seed derived from the text, split, and augmentation
seed makes this view deterministic. Key and query texts are encoded
separately. AG News therefore tests transfer to a different text-representation
distribution, rather than category prediction.

Preprocessing collapses whitespace, applies word-count limits, and removes
exact text duplicates across splits. WikiText uses its supplied train,
validation, and test splits. AG News holds out validation texts from its
training split using a deterministic hash partition and retains the official
test split. Sample limits and seeds are defined in the training programs.

No datasets, model weights, or representation caches are distributed here.
The first run needs network access until the public inputs have been cached.
Run artifacts are written beneath `HEAD_GEOMETRY_ROOT` (default `outputs/`);
the download libraries use their configured caches. See
[reproducibility](reproducibility.md) for experiment paths and cache reuse.

# Synonym Recognition by Embedding

![Language](https://img.shields.io/github/languages/top/hanzo89757/Synonym-Recognition-by-Finetune-Embedding.svg?style=flat)
![Licence](https://img.shields.io/github/license/hanzo89757/Synonym-Recognition-by-Finetune-Embedding.svg?style=flat)

## Relation Resource

 - Synonym dataset `datasets/synonyms/*` is built on Chinese Synonym Dataset: [同义词词林](https://www.ltp-cloud.com/download#down_cilin).
 - Pre-train word embedding: 
    1. [Word2vec or Fasttext](https://github.com/Kyubyong/wordvectors)
    2. [Wikipedia2vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained)
    3. [Tencent AI Lab Embedding Corpus for Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/embedding.html)
    4. ···
    
## Get Started

### Prepare for synonym dataset

You can use `datasets/synonyms/*` or dataset else you built.

### Download the embedding

Download from the above Pre-train word embedding.

### Dependencies

You can install dependencies by:

```shell
python install -r requirements.txt
```

### Run

```shell 
python main.py --train datasets/synonyms/train
               --dev datasets/synonyms/dev \
               --test datasets/synonyms/test \
               --embedding /path/to/embedding_file \
               --outputs /path/to/outputs_dir
```

## License
@Apache 2.0 (Except for `datasets`)

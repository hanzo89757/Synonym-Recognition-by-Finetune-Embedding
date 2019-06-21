# -*- coding:utf-8 -*-

import os

EMBEDDING_DIR = 'datasets/embeddings'
BATCH_SIZE = 512


def eval_embed(embedding_name):
    print('Start to process {} ...'.format(embedding_name))
    os.system(
        """
        python main.py --train datasets/synonyms/train \
                       --dev datasets/synonyms/dev \
                       --test datasets/synonyms/test \
                       --embedding {1}/{0} \
                       --outputs outputs_fine_tune/{0}_{2} \
                       --log-level debug \
                       --batch-size {2} \
                       --log logs/{0}_{2}.log
        """.format(embedding_name, EMBEDDING_DIR, BATCH_SIZE)
    )
    print('Finish {}.'.format(embedding_name))


if __name__ == '__main__':
    for file in os.listdir(EMBEDDING_DIR):
        eval_embed(file)

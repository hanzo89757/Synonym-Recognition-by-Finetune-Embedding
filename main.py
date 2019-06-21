# -*- coding:utf-8 -*-

import argparse
import logging

import torch
import torch.nn as nn

from config import device
from models.mlp import LinearNet
from utils.evolutor import Evaluator
from utils.io import rebuild_dir
from utils.preprocess import PreProcess
from utils.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store', dest='train', default='datasets/examples',
                        help='The dir of train data.')
    parser.add_argument('--dev', action='store', dest='dev', default='datasets/examples',
                        help='The dir of dev data.')
    parser.add_argument('--test', action='store', dest='test', default='datasets/examples',
                        help='The dir of test data')
    parser.add_argument('--embedding', action='store', dest='embedding',
                        default='datasets/examples/embedding.txt',
                        help='Pre-trained word embedding.')
    parser.add_argument('--outputs', action='store', dest='outputs', default='outputs/default',
                        help='Dir of intermediate files.')
    parser.add_argument('--epoch', action='store', dest='epoch', default=10, type=int,
                        help='Epoch.')
    parser.add_argument('--batch-size', action='store', dest='batch_size', default=64, type=int,
                        help='Batch size.')
    parser.add_argument('--log', dest='log',
                        help='Logging filename.')
    parser.add_argument('--log-level', dest='log_level', default='info',
                        help='Logging level.')

    opts = parser.parse_args()

    # mkdir outputs if not dir
    rebuild_dir(opts.outputs)

    # logger configure
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, filename=opts.log, level=getattr(logging, opts.log_level.upper()))
    logging.info(opts)

    # pre-process dataset
    pre_process = PreProcess(embedding_path=opts.embedding)
    train_pos_dataset, _ = pre_process.load_data(opts.train)
    dev_dataset = pre_process.load_data(opts.dev)
    test_dataset = pre_process.load_data(opts.test)

    pre_process.build_vocab(*([train_pos_dataset] + dev_dataset + test_dataset))
    synonym_dict = pre_process.build_synonym_dict(train_pos_dataset)

    # pre-trained embedding
    vectors = pre_process.words_field.vocab.vectors
    embedding = nn.Embedding(*vectors.size())
    embedding.weight = nn.Parameter(vectors, requires_grad=False)
    embedding.to(device)

    # create model
    _, dim = vectors.size()
    model = LinearNet(dim, dim * 2, dim * 2, dim)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # train
    trainer = Trainer(embedding=embedding, outputs_dir=opts.outputs,
                      batch_size=opts.batch_size, epoch=opts.epoch)
    trainer.train(model, train_pos_dataset, synonym_dict)

    # evaluate
    evaluator = Evaluator(embedding=embedding, outputs_dir=opts.outputs)
    threshold = evaluator.evaluate(*dev_dataset, model=model)
    evaluator.evaluate(*test_dataset, model=model, threshold=threshold)

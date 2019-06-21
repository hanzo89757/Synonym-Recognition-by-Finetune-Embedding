# -*- coding:utf-8 -*-

import logging
import os
import pickle

import torch
import torchtext
from torchtext.vocab import Vectors

from config import device
from utils.io import rebuild_dir


class PreProcess(object):
    """pre-process the synonym words and embedding"""
    words_field = torchtext.data.Field(sequential=True,
                                       batch_first=True,
                                       pad_token=None,
                                       unk_token=None,
                                       tokenize=(lambda s: s.split(',')))

    def __init__(self, embedding_path, cached_dir='.cached'):
        self.vectors = Vectors(embedding_path)
        logging.info('The dimension of pre-trained word embedding is {}'.format(self.vectors.dim))

        self._embedding_name = os.path.basename(embedding_path)
        self._cached_dir = cached_dir
        rebuild_dir(cached_dir)

    def load_data(self, dataset_dir):
        """
        load dataset from directory
        :param dataset_dir: dir of dataset (contains pos.csv and neg.csv)
        :return: [dataset_pos, dataset_neg]
        """

        datasets = []
        for filename in ['pos.csv', 'neg.csv']:
            file_path = os.path.join(dataset_dir, filename)
            dataset = torchtext.data.TabularDataset(file_path,
                                                    fields=[('words', self.words_field)],
                                                    format='tsv')
            logging.info('Dataset {}:'.format(file_path))
            logging.info('\tTotal Length: {}'.format(len(dataset.examples)))
            self._filter_not_existed_embedding(dataset, self.vectors.stoi)
            logging.info('\tExisted in embedding Length: {}'.format(len(dataset.examples)))

            if not len(dataset.examples):
                logging.error('There is no available example in {}'.format(file_path))
                exit()

            datasets.append(dataset)
        return datasets

    def build_vocab(self, *datasets):
        logging.info('Start building vocabulary...')
        self.words_field.build_vocab(*datasets, vectors=self.vectors)
        logging.info('Finish building vocabulary...')
        logging.info('Vocabulary Length: {}'.format(len(self.words_field.vocab)))

    def build_synonym_dict(self, dataset):
        synonym_path = os.path.join(self._cached_dir, '{}_synonym_dict.pkl'.format(self._embedding_name))

        if os.path.exists(synonym_path):
            logging.info('Load synonym dictionary from cached {}'.format(synonym_path))
            with open(synonym_path, 'rb') as handle:
                synonym_dict = pickle.load(handle)
        else:
            logging.info('Start building antonym dictionary...')
            stoi = self.words_field.vocab.stoi
            synonym = ([stoi[s] for s in example.words]  # [(left_id, right_id), ...]
                       for example in dataset.examples)

            synonym_dict = {}
            for (word_left, word_right) in synonym:
                synonym_dict.setdefault(word_left, []).append(word_right)
                synonym_dict.setdefault(word_right, []).append(word_left)

            for word in synonym_dict:
                synonym_dict[word] = torch.tensor(synonym_dict[word], dtype=torch.int64, device=device)
            logging.info('Finish building antonym dictionary...')

            with open(synonym_path, 'wb') as handle:
                pickle.dump(synonym_dict, handle)
            logging.info('Save synonym dictionary to {}'.format(synonym_path))
        return synonym_dict

    @staticmethod
    def _filter_not_existed_embedding(dataset, stoi):
        dataset.examples = list(filter(
            lambda x: x.words[0] in stoi and x.words[1] in stoi, dataset.examples
        ))

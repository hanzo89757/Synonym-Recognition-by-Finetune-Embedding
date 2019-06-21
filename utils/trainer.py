# -*- coding:utf-8 -*-

import logging
import os

import torch
import torchtext
from tqdm import tqdm

from config import device


class Trainer(object):
    """Train linear net using positive train dataset"""

    def __init__(self, embedding, outputs_dir, batch_size=64, epoch=10, attract_margin=0.6):
        self._embedding = embedding
        self._batch_size = batch_size
        self._epoch = epoch
        self._attract_margin = attract_margin
        self._outputs_dir = outputs_dir

        self._cos = torch.nn.CosineSimilarity(dim=1)
        self._optimizer = None

    def train(self, model, dataset, synonym_dict):
        """
        train model by positive dataset and antonym dictionary
        :param model:
        :param dataset:
        :param synonym_dict:
        :return:
        """
        model.train(True)

        self._optimizer = torch.optim.Adam(model.parameters())
        for i in range(self._epoch):

            logging.info('Training...  Epoch: {}'.format(i))
            loss = self.train_epoch(model, dataset, synonym_dict, i)
            logging.info('\tLoss: {}'.format(loss))

    def train_epoch(self, model, dataset, synonym_dict, epoch=0):
        batch_iter = torchtext.data.BucketIterator(dataset=dataset,
                                                   batch_size=self._batch_size,
                                                   shuffle=True,
                                                   device=device)
        loss_all = 0.

        pbar = tqdm(total=len(dataset), ascii=True, desc='[Epoch {}] Train'.format(epoch))
        for batch in batch_iter:
            words = batch.words
            init_embedded_words = self._embedding(words)
            embedded_words = model(init_embedded_words)

            loss = 0.
            for i in range(len(init_embedded_words)):
                loss += self.loss(embedded_words[i][0],
                                  embedded_words[i][1],
                                  init_embedded_words[i][0],
                                  init_embedded_words[i][1],
                                  synonym_dict[words[i][0].item()],
                                  synonym_dict[words[i][1].item()])
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            loss_all += loss

            pbar.update(self._batch_size)
        pbar.close()

        model_save_path = os.path.join(self._outputs_dir, 'models.pkl')
        torch.save(model, model_save_path)

        return loss_all.item()

    def loss(self, left_emb, right_emb, left_init_emb, right_init_emb,
             left_synonyms, right_synonyms):
        """
        :param left_emb:
        :param right_emb:
        :param left_init_emb:
        :param right_init_emb:
        :param left_synonyms:
        :param right_synonyms:
        :return:
        """
        embedding_weight = self._embedding.weight

        left_cos = self._cos(left_emb.expand_as(embedding_weight), embedding_weight)
        left_cos[left_synonyms] = -1
        _, left_closest_index = torch.max(left_cos, dim=0)

        right_cos = self._cos(right_emb.expand_as(embedding_weight), embedding_weight)
        right_cos[right_synonyms] = -1
        _, right_closest_index = torch.max(right_cos, dim=0)

        left_closest_antonym = embedding_weight[left_closest_index.item()]
        right_closest_antonym = embedding_weight[right_closest_index.item()]

        loss_left = max(torch.dot(left_closest_antonym, left_emb) -
                        torch.dot(left_emb, right_emb) + self._attract_margin, 0)
        loss_right = max(torch.dot(right_closest_antonym, right_emb) -
                         torch.dot(left_emb, right_emb) + self._attract_margin, 0)
        loss_base = torch.norm(left_init_emb - left_emb, 2) + torch.norm(right_init_emb - right_emb, 2)

        loss = loss_left + loss_right + loss_base
        return loss

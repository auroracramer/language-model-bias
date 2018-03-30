# coding: utf-8

# In[5]:


import os
import torch
import preprocess
import itertools


# In[2]:


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


# In[7]:


class Corpus(object):
    def __init__(self, path, vocab, idx_train, idx_val, idx_test):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path, vocab, idx_train)
        self.valid = self.tokenize(path, vocab, idx_val)
        self.test = self.tokenize(path, vocab, idx_test)
        
    def tokenize(self, path, vocab, idx):
        tokens = 0
        for ind in idx['id'].values:
            article = preprocess.read_preprocessed_file(os.path.join(path, ind), vocab)
            words = list(itertools.chain.from_iterable(article)) + ['<eos>']
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word)
        
        ids = torch.LongTensor(tokens)
        token = 0
        for ind in idx['id'].values:
            article = preprocess.read_preprocessed_file(os.path.join(path, ind), vocab)
            words = list(itertools.chain.from_iterable(article)) + ['<eos>']
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                token += 1
                
        return ids
import os
import re

import pickle
import numpy as np

from scipy.sparse import csr_matrix, save_npz, load_npz
from collections import Counter

from tqdm import tqdm as tqdm_notebook
from functools import lru_cache

## "Внутренние" импорты
from utils import two_way_map

log = open('Log.txt','w',encoding='utf-8')

class NGramMatrix:
    def __init__(self, n=2, tokenizer=None):
        # Максимальная длина N-граммы
        self.n = n
        self.tokenizer = tokenizer
        # Длина корпуса в токенах
        self.N = 0
        self.counts = dict()
        self.ngram_matrix = csr_matrix([])
        self.id2key, self.key2id = list(), dict()
    
    def save(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        for key,val in self.__dict__.items():
            if type(val) == csr_matrix:
                save_npz(os.path.join(directory, f"{key}.npz"), val)
            # if type(val) == dict or type(val) == list:
            else:
                with open(os.path.join(directory, f"{key}.pickle"),'wb') as outp:
                    # if type(val) == dict:
                    #     json.dump({str(a):b for a,b in val.items()}, outp, ensure_ascii=False)
                    # elif type(val) == list:
                    #     json.dump(val, outp, ensure_ascii=False)
                    pickle.dump(val, outp)
            
    @classmethod
    @lru_cache(maxsize=3)
    def load(cls, directory):
        matrix = cls.__new__(cls)
        for file in os.listdir(directory):
            fname, ext = file.split('.')
            if ext == 'pickle':
                with open(os.path.join(directory,file),'rb') as inp:
                    matrix.__setattr__(fname,pickle.load(inp))
            elif ext == 'npz':
                matrix.__setattr__(fname, load_npz(os.path.join(directory, file)))
        return matrix

    def train(self, corpus, verbose=True):
        print('Tokenizing...')

        if type(corpus) == str:
            corpus = self.tokenizer(corpus)
        else:
            corpus = [sent for text in tqdm_notebook(corpus,
            total=len(corpus))for sent in self.tokenizer(text)]

        self.N = sum(len(sent) for sent in corpus)

        ## Подсчитываем частоты униграмм
        self.counts = Counter(word for sent in corpus for word in sent)
        n_cols = len(self.counts)
        n_rows = n_cols

        ## Подсчитывем частоты N-грамм с N>1:
        for i in range(2, self.n+1):
            self.counts += Counter(tuple(sent[j:j+i]) for sent in corpus \
                for j in range(len(sent)-i))
            if i == self.n-1:
                n_rows = len(self.counts)

        self.id2key, self.key2id = two_way_map(self.counts)
        values, row_ids, col_ids = np.zeros(len(self.counts)-n_cols),\
                                   np.zeros(len(self.counts) - n_cols),\
                                   np.zeros(len(self.counts)-n_cols)
        i = 0

        print('Calculating n-gram frequencies...')

        if verbose:
            count_iter = tqdm_notebook(self.counts, total=len(self.counts))

        for key in count_iter:
            if type(key) == tuple:
                values[i] = self.counts[key]
                if len(key[:-1]) == 1:
                    row_ids[i] = self.key2id[key[0]]
                else:
                    row_ids[i] = self.key2id[key[:-1]]
                col_ids[i] = self.key2id[key[-1]]
                i += 1

        self.ngram_matrix = csr_matrix((values, (row_ids, col_ids)),
                                        shape=(n_rows, n_cols))
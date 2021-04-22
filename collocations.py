import numpy as np

from scipy.sparse import csr_matrix

from tqdm import tqdm as tqdm_notebook

## "Внутренние" импорты
from .ngrams import NGramMatrix

class CollocationMetric:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, ngram_matrix, verbose):
        values, row_ids, col_ids = np.zeros(len(ngram_matrix.ngram_matrix.data)), \
                                   np.zeros(len(ngram_matrix.ngram_matrix.data)), \
                                   np.zeros(len(ngram_matrix.ngram_matrix.data))
        i = 0

        if verbose:
            item_iter = ngram_matrix.counts
        else:
            item_iter = tqdm_notebook(ngram_matrix.counts,
            total=len(ngram_matrix.counts))
        
        for ab in item_iter:
            if type(ab) == tuple:
                if len(ab[:-1]) == 1:
                    elem_a = ab[0]
                else:
                    elem_a = ab[:-1]
                elem_b = ab[-1]
                values[i] = self.metric(ab=ngram_matrix.counts[ab],
                                        a=ngram_matrix.counts[elem_a],
                                        b=ngram_matrix.counts[elem_b],
                                        N=ngram_matrix.N)
                row_ids[i] = ngram_matrix.key2id[elem_a]
                col_ids[i] = ngram_matrix.key2id[elem_b]
                i += 1
        return csr_matrix((values, (row_ids, col_ids)),
                          shape=ngram_matrix.ngram_matrix.shape)

class PMI(CollocationMetric):
    def __init__(self):
        pmi = lambda ab,a,b,N: np.log(ab) + np.log(N) - np.log(a) - np.log(b)
        super().__init__(pmi)

func_POS = ['PART','CCONJ', 'SCONJ', 'ADP', 'AUX', 'DET','PRON','PUNCT','NUM']

class CollocateMatrix(NGramMatrix):
    def __init__(self, n, tokenizer, metric):
        super().__init__(n, tokenizer)
        self.metric = metric
        self.collocate_matrix = None

    def train(self, corpus):
        print('Training N-gram matrix...')
        super().train(corpus)
        print('Calculating metric...')
        self.collocate_matrix = self.metric(self, verbose=True)

    def get_collocations(self, text, thresh=1.0,
    tokenize=False, no_function_words=True):

        if tokenize:
            text = [word for sent in self.tokenizer(text) for word in sent]
        outp = []

        for ngram_range in range(2, self.n+1):
            for token_id in range(len(text)-ngram_range+1):
                if ngram_range == 2:
                    key = text[token_id]
                else:
                    key = tuple(text[token_id:token_id+ngram_range-1])
                val = text[token_id+ngram_range-1]

                ## Filter function words:
                if no_function_words:
                    if val.split('_')[-1] in func_POS:
                        continue

                strength = self.collocation_strength(key, val)

                if ngram_range == 2:
                    key = (key,)
                if strength is not None:
                    if strength >= thresh:
                        outp.append((
                            (token_id, token_id+ngram_range-1),
                            (key, val),
                            strength
                        ))
        

        outp = sorted(outp, key=lambda x: x[0])
        return outp
    
    def collocation_strength(self, key, val):
        if key in self.key2id and val in self.key2id:
            key_id, val_id = self.key2id[key], self.key2id[val]
            strength = self.collocate_matrix[key_id].toarray().squeeze()[val_id]
            return strength
        else:
            return None

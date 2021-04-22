import argparse
import sys

import numpy as np
import pandas as pd
import torch as tt

from itertools import chain
from functools import lru_cache

from abc import ABC, abstractmethod
from copy import copy

from gensim.models import KeyedVectors
from gensim import matutils

from transformers import BertForMaskedLM, BertTokenizer

from tqdm import tqdm

## Local imports:
from .collocations import CollocateMatrix
from .preprocessing import Tokenizer
from .nn_utils import UnsupervisedBatchIterator, make_predictions
from .utils import inflect, last_underscore_split

## TO DO:
## Встраивание предсказанного слова в предложение в GensimCollocateReplacer- Done
## - Интерфейс командной строки - загрузка из excel/csv с вышрузкой в excel/csv с нужной комбинацией моделей - In progress
## - Добавить интерфейс для обучения коллокационной матрицы и bert'а - если успею
## - Другие модели выбора слова для замены (вся коллокация, именованная сущность, рандомное слово) - если успею
## - Обучить коллокационную матрицу на cased data - если успею
## - Зафайнтюнить берт на датасете юмора с маскированием всех не-служебных слов

class TokenSelector(ABC):
    '''
    Выбирает индексы токенов
    в предложении, которые нужно заменить
    '''

    @abstractmethod
    def select_tokens(self, sents):
        '''
        Возвращает список списков кортежей:
        Для каждого предложения
        Список индексов промежутков в нём,
        которые нужно заменить
        '''
        pass

class CollocateTokenSelector(TokenSelector):
    def __init__(self, colloc_matrix_path, thresh=2.0, direction='right', verbose=False, **kwargs):
        '''
        Выбрать слова для замены в устойчивых коллокациях

        colloc_matrix_path - Путь к сохранённой коллокационной матрице
        '''
        self.cmatrix = CollocateMatrix.load(colloc_matrix_path)
        self.thresh = thresh
        self.direction = direction
        self.verbose = verbose
    
    def select_tokens(self, sents, mask_token='[MASK]'):
        ## Токенизируем, предполагая,
        ## что один заголовок - всегда одно предложение
        
        tokenized_sents = []
        masked_sents = []
        replacement_ids = []
        replaced_words = []

        if self.verbose:
            print("Selecting collocates to replace")
            iterator = tqdm(sents, total=len(sents))
        else:
            iterator = sents

        for sent in iterator:
            sent, sent_tokens = self.cmatrix.tokenizer(sent, include_orig_tokens=True)

            sent = [word for s in sent for word in s]
            sent_tokens = [word[:word.rfind('_')] for s in sent_tokens for word in s]

            sent_repl_ids = []
            sent_replaced_words = []
            masked_variants = []

            collocations = self.cmatrix.get_collocations(sent,
            thresh=self.thresh, tokenize=False)

            for idxs, collocation, strength in collocations:
                ## Берём правую часть коллокации (всегда одно слово)
                repl_id = idxs[-1]

                if (repl_id,) not in sent_repl_ids:
                    sent_repl_ids.append((repl_id,))
                    sent_replaced_words.append((collocation[-1],))

                    masked_variant = copy(sent_tokens)
                    masked_variant[repl_id] = mask_token
                    masked_variants.append(' '.join(masked_variant))
            
            tokenized_sents.append(sent)
            masked_sents.append(masked_variants)
            replacement_ids.append(sent_repl_ids)
            replaced_words.append(sent_replaced_words)
        
        return tokenized_sents, replacement_ids, replaced_words, masked_sents
                
class TokenReplacer(ABC):
    pass

class TokenIDReplacer(TokenReplacer, ABC):
    @abstractmethod
    def replace_tokens(self, sents, replacement_ids):
        pass

class MaskedTokenReplacer(TokenReplacer, ABC):
    @abstractmethod
    def replace_tokens(self, masked_sentences):
        pass

class GensimCollocateReplacer(TokenIDReplacer):
    def __init__(self, gensim_model_path: str, colloc_matrix_path: str, dist_thresh=0.6, colloc_thresh=2.0, verbose=False, **kwargs):
        self.gensim_model = KeyedVectors.load_word2vec_format(gensim_model_path, binary=True)
        self.cmatrix = CollocateMatrix.load(colloc_matrix_path)
        self.dist_thresh = dist_thresh
        self.colloc_thresh = colloc_thresh
        self.verbose = verbose
    
    def most_distant_to_vec(self, word_vec,
    pos, max_count=10):
        to_dist = lambda x: 1 - (x + 1)/2
        dists = np.dot(self.gensim_model.vectors, word_vec)
        ## map cosines to (0,1) range
        ## where higher values indicate higher distance:
        dists = to_dist(dists)

        sorted_dist_ids = matutils.argsort(dists, reverse=True)
        
        word_distances = [
            (self.gensim_model.index2entity[word_id], float(dists[word_id]))
            for word_id in sorted_dist_ids \
                if self.gensim_model.index2entity[word_id].endswith(pos)
                    and float(dists[word_id]) > self.dist_thresh
        ]

        if max_count:
            word_distances = word_distances[:max_count]
        
        return word_distances
    
    def replace_tokens(self, sents, replacement_ids):
        changed_headlines = []
        new_words = []

        if self.verbose:
            print("Replacing words with the most semantically distinct collocates of their neighbours")
            iterator = tqdm(zip(sents, replacement_ids), total=len(sents))
        else:
            iterator = zip(sents, replacement_ids)

        for sent, to_replace in iterator:
            sent_changed_headlines = []
            sent_new_words = []

            for variant in to_replace:
                changed_headline = sent
                new_word = None

                span_start_id = variant[0]
                span_end_id = variant[-1]+1
                
                value = sent[span_start_id:span_end_id]

                ## Берём часть речи первого слова в span'е
                pos = value[0].split('_')[1]
                word_vector = np.mean(np.array([self.gensim_model[word]\
                    for word in sent[span_start_id:span_end_id]\
                    if word in self.gensim_model]),
                axis=0)

                if type(word_vector) == np.ndarray:
                    most_distant = self.most_distant_to_vec(word_vector,
                                                            pos)

                    for new_word_id, (new_word, distance) in enumerate(most_distant):
                        cstrength = 0

                        ## Select strongest collocation key:
                        for i in range(1, self.cmatrix.n):
                            key = tuple(sent[span_start_id-i:span_start_id])
                            cstrength_key = self.cmatrix.collocation_strength(key, new_word)
                            if cstrength_key is not None:
                                if cstrength_key > cstrength:
                                    cstrength = cstrength_key
                        
                        most_distant[new_word_id] = (new_word, distance, cstrength)
                    
                    ## Выберем наилучшего кандидата на замену данного слова
                    # Прибавляем единицу на случай если коллокационная матрица не знает такого сочетания,
                    # чтобы ранжировать только по расстоянию:
                    candidates = [(w,d*(cs+1)) for w,d,cs in most_distant]

                    new_word = max(candidates, key=lambda x: x[1])[0]

                    changed_headline = sent[:span_start_id] + [new_word] + sent[span_end_id:]
                sent_changed_headlines.append(changed_headline)
                sent_new_words.append(new_word)

            changed_headlines.append(sent_changed_headlines)
            new_words.append(sent_new_words)
        
        return changed_headlines, new_words

class BERTReplacer(MaskedTokenReplacer):
    def __init__(self, bert_model_path: str, bert_tokenizer: str,
                 verbose=False,
                 **kwargs):
        ## Загрузим базовую модель и токенизатор для неё:
        self.bert_model = tt.load(bert_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.verbose = verbose

    def replace_tokens(self, masked_sentences):
        ## Превращаем двумерный список вариантов изменения каждого предложения в одномерный:
        masked_sentences_flat = list(chain(*masked_sentences))
        batch_iter = UnsupervisedBatchIterator(masked_sentences_flat)
        changed_headlines_flat, new_words_flat = make_predictions(batch_iter, self.bert_model,
        self.tokenizer, self.verbose)

        ## "Заворачиваем" обратно в двумерный список:
        changed_headlines_iter, new_words_iter = iter(changed_headlines_flat), iter(new_words_flat)
        changed_headlines, new_words = [], []
        for sent in masked_sentences:
            changed_headline_variants = []
            new_words_sent = []
            for i in range(len(sent)):
                changed_headline_variants.append(next(changed_headlines_iter))
                new_words_sent.append(next(new_words_iter))
            changed_headlines.append(changed_headline_variants)
            new_words.append(new_words_sent)

        return changed_headlines, new_words

class Humourizer:
    def __init__(self,
                 token_selector: TokenSelector,
                 token_replacer: TokenReplacer,
                 verbose: bool):
        ## Используем инъекцию зависимостей:
        ## То как именно будут процессится
        ## предложения зависит от такого, какие
        ## объекты будут занимать поля
        ## token_selector,
        ## token_replacer
        self.token_selector = token_selector
        self.token_replacer = token_replacer
        self.verbose = True
    
    def join_new_sentences_upos_(self, headlines, replacement_ids, new_words):
        output_sentences = []

        for sent, sent_repl_ids, sent_new_words in zip(headlines,
        replacement_ids, new_words):
            sent_variants = []

            sent_lemmas, sent_tokens = self.token_replacer.cmatrix.tokenizer(sent,
            include_orig_tokens=True)

            sent_lemmas = [word for s in sent for word in s]
            sent_tokens = [word for s in sent_tokens for word in s]

            for replacement_id, new_word in zip(sent_repl_ids, sent_new_words):
                if new_word:
                    lemma, upos = last_underscore_split(new_word)

                    ## Форму (пока) определяем по первому токену заменённого промежутка
                    orig_token, orig_xpos = last_underscore_split(sent_tokens[replacement_id[0]])

                    new_word_flexed = inflect(lemma, orig_xpos)

                    ## "Собираем" предложение обратно
                    tokens = [i[:i.rfind('_')] for i in sent_tokens]
                    sent_variant = tokens[:replacement_id[0]]
                    sent_variant += [new_word_flexed]
                    sent_variant += tokens[replacement_id[-1]+1:]
                    sent_variant = ' '.join(sent_variant)
                else:
                    sent_variant = sent

                sent_variants.append(sent_variant)
            
            output_sentences.append(sent_variants)
        return output_sentences
    
    def vandalize_headlines(self, headlines, return_pandas=False):
        ## Выбираем слова для замены
        tokenized_headlines, replacement_ids, replaced_words, masked_headlines = self.token_selector.select_tokens(headlines)
        # ## Маскируем слова для замены (для наглядности)
        # masked_sentences, masked_words = self.token_selector.mask_sentences(self, headlines, replacement_ids)
        ## Заменяем слова
        if isinstance(self.token_replacer, TokenIDReplacer):
            changed_headlines, new_words = self.token_replacer.replace_tokens(tokenized_headlines, replacement_ids)
            ## собираем предложение обратно:
            if type(self.token_replacer) == GensimCollocateReplacer:
                changed_headlines = self.join_new_sentences_upos_(headlines, replacement_ids, new_words)

        elif isinstance(self.token_replacer, MaskedTokenReplacer):
            changed_headlines, new_words = self.token_replacer.replace_tokens(masked_headlines)

        
        if return_pandas:
            outp = []

            for sent_id in range(len(tokenized_headlines)):
                for masked, index, replaced, changed, new in zip(masked_headlines[sent_id],
                  replacement_ids[sent_id],
                  replaced_words[sent_id],
                  changed_headlines[sent_id],
                  new_words[sent_id]):
                    outp.append({
                        'headline':headlines[sent_id],
                        'masked': masked,
                        'tokenized': tokenized_headlines[sent_id],
                        'span_index': index,
                        'span': replaced,
                        'predicted': changed,
                        'new span': new
                    })
            
            outp = pd.DataFrame(outp)
            return outp
                    
        return (tokenized_headlines, masked_headlines, replacement_ids,
        replaced_words, changed_headlines, new_words)


def test_collocate_gensim():
    token_selector = CollocateTokenSelector(colloc_matrix_path='CM_SpaCy_proper1')
    token_replacer = GensimCollocateReplacer(colloc_matrix_path='CM_SpaCy_proper1',
                                            gensim_model_path='gensim_models/udpipe_wikipedia/model.bin')
    sentences = ['i saw donald trump on wall street',
    'great britain to announce new prime minister']
    humourizer = Humourizer(token_selector, token_replacer, verbose=False)
    return humourizer.vandalize_headlines(sentences,
    return_pandas=True)

def test_collocate_bert():
    token_selector = CollocateTokenSelector(colloc_matrix_path='CM_SpaCy_proper1')
    token_replacer = BERTReplacer('bert_masked_lm_full_model',
                                  'bert-large-uncased-whole-word-masking')
    sentences = ['i saw donald trump on wall street',
    'great britain to announce new prime minister']
    humourizer = Humourizer(token_selector, token_replacer, verbose=False)
    return humourizer.vandalize_headlines(sentences,
    return_pandas=True)

def main(input_file: str, output_file: str, verbose: bool,
         keep_case: bool, token_selector: str, token_replacer: str,
         **kwargs):
    ext_in = input_file[input_file.rfind('.')+1:].lower()
    ext_out = output_file[output_file.rfind('.')+1:].lower()

    if ext_in not in ('csv', 'xlsx'):
        raise ValueError(f"Input extension .{ext_in} not supported")
    
    if ext_out not in ('csv', 'xlsx'):
        raise ValueError(f"Output extension .{ext_out} not supported")
    
    if ext_in == 'csv':
        df1 = pd.read_csv(input_file)
    elif ext_out == 'xlsx':
        df1 = pd.read_excel(input_file, engine='openpyxl')
    
    df1 = df1.dropna(subset=['headline'])

    if not keep_case:
        df1['headline'] = df1['headline'].str.lower()
    
    headlines = df1['headline']
    
    token_selector_cls = getattr(sys.modules[__name__], token_selector)
    token_replacer_cls = getattr(sys.modules[__name__], token_replacer)

    token_selector = token_selector_cls(**kwargs, verbose=verbose)
    token_replacer = token_replacer_cls(**kwargs, verbose=verbose)

    humourizer = Humourizer(token_selector, token_replacer, verbose)

    df2 = humourizer.vandalize_headlines(headlines, return_pandas=True)

    df_out = df1.merge(df2, on='headline', how='left')

    if ext_out == 'csv':
        df_out.to_csv(output_file)
    elif ext_out == 'xlsx':
        df_out.to_excel(output_file, engine='openpyxl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Vandalize news headlines by making humorous edits of them.
    News headlines should be contained in a column called 'headline' of a CSV or XLSX file.
    The current implementation outputs all possible variants produced by algorithms''')

    parser.add_argument('input_file', help='CSV or Excel file with string column headline')
    parser.add_argument('output_file', help='Path to CSV/Excel file with headline_vandalizer output')

    parser.add_argument('--silent', dest='verbose', action='store_false',
    help='Not to display information regarding progress of data processing')
    parser.add_argument('--keep_case', dest='keep_case', action='store_true',
    help='Do not lowercase input data')

    # подклассы TokenSelector - конкретные реализации алгоритмов выбора токенов:
    token_selector_list = [c.__name__ for c in TokenSelector.__subclasses__()]
    # только подклассы подклассов TokenReplacer - конкрентные реализации алгоритмов выбора токенов:
    token_replacer_list = [subcl.__name__ for c in TokenReplacer.__subclasses__() for subcl in c.__subclasses__()]

    ## Model-specific args:
    parser.add_argument('--word_selector', dest='token_selector', default='CollocateTokenSelector', choices=token_selector_list,
    help="Which word selection algorithm to use")
    parser.add_argument('--word_replacer', dest='token_replacer', default='GensimCollocateReplacer', choices=token_replacer_list,
    help="Which word replacement algorithm to use")

    parser.add_argument('--gensim_model_path', dest='gensim_model_path', default='gensim_models/udpipe_wikipedia/model.bin',
    help='Path to gensim model file (must be ib binary Word2Vec format). Required if you are going to use GensimCollocateReplacer')
    parser.add_argument('--colloc_matrix_path', dest='colloc_matrix_path', default='CM_SpaCy_proper1',
    help='Path to folder with saved CollocateMatrix. Required if you are going to use CollocateTokenSelector and/or GensimCollocateReplacer')

    parser.add_argument('--bert_model_path', dest='bert_model_path', default='bert_masked_lm_full_model',
    help="Path to saved BERT model. Required if you are going to use BERTReplacer")
    parser.add_argument('--bert_tokenizer', dest='bert_tokenizer', default='bert-large-uncased-whole-word-masking',
    help='Name of BERT tokenizer from transformers repository which was used for model training. Required if you are going to use BERTReplacer')

    args = vars(parser.parse_args())

    main(**args)

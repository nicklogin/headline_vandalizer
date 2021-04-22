import numpy as np

from .word_change import most_distant_same_pos

def replace_collocates(s, cm, wv_model, dist_thresh=0.65, colloc_thresh=2.0):
    '''Провести юморизацию предложения
    s - строка, предложения на вход
    cm - коллокационная матрица
    wv_model - генсимовская модель
    tokenizer - токенизатор
    '''
    collocations = cm.get_collocations(s, thresh=colloc_thresh)
    collocations = sorted(collocations, key=lambda x: x[2], reverse=False)
    output = []
    
    for colloc in collocations:
        word_a, word_b = colloc[1]
        
        if not word_a.split('_')[-1] in ('PART','CCONJ', 'SCONJ', 'ADP','AUX','DET','PRON','PUNCT','NUM') and\
        not word_b.split('_')[-1] in ('PART','CCONJ', 'SCONJ', 'ADP', 'AUX', 'DET','PRON','PUNCT','NUM'):
            pass
        else:
            continue
        
        try:
            candidates = most_distant_same_pos(word_b, wv_model, thresh=dist_thresh)
        except KeyError:
            continue
        for candidate, candidate_dist in candidates:
            strength = cm.collocation_strength(word_a, candidate)
            if strength is not None:
                if strength > colloc_thresh:
                    output.append((colloc[0], (word_a, word_b, colloc[2]), (word_a, candidate, strength),
                                  candidate_dist))
        
    return output

def replace_words(s, cm, wv_model, tokenizer, dist_thresh=0.65, colloc_thresh=2.0):
    '''Провести юморизацию предложений
    s - массив строк, предложения на вход
    cm - коллокационная матрица
    wv_model - генсимовская модель
    tokenizer - токенизатор
    '''
    output = []
    ## токенизируем предлоения
    sents = tokenizer(s)
    for tokens in sents:
        ## Сформировать биграммы в предложении
        word_pairs = [((i,j), (tokens[i],tokens[j]), cm.collocation_strength(tokens[i],tokens[j])) for i,
                      j in zip(list(range(len(tokens))),list(range(1,len(tokens))))]

        for colloc in word_pairs:
            word_a, word_b = colloc[1]

            ## отсеиваем коллокации, содержащие служебные слова
            if not word_a.split('_')[-1] in ('PART','CCONJ', 'SCONJ', 'ADP','AUX','DET','PRON','PUNCT','NUM') and\
            not word_b.split('_')[-1] in ('PART','CCONJ', 'SCONJ', 'ADP', 'AUX', 'DET','PRON','PUNCT','NUM'):
                pass
            else:
                continue

            ## находим кандидатов на замену
            try:
                candidates = most_distant_same_pos(word_b, wv_model, thresh=dist_thresh)
            except KeyError:
                continue
            for candidate, candidate_dist in candidates:
                ## сравниваем коллокационную силу кандидатов
                strength = cm.collocation_strength(word_a, candidate)
                if strength is not None:
                    if strength > colloc_thresh:
                        output.append((colloc[0], (word_a, word_b, colloc[2]), (word_a, candidate,strength),
                                      candidate_dist))
        
    return output

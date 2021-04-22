from gensim import matutils

from gensim.models import KeyedVectors

import numpy as np


def most_distant_same_pos(word, model,
thresh=0, max_count=0, pos_after_word=True):
    '''
    Find most distant words to given
    assuming the model is binary (and therefore normalized)
    and includes pos-tagging (in form of word_POS)
    '''
    to_dist = lambda x: 1 - (x + 1)/2
    word_vec = matutils.unitvec(model[word]).astype(np.float32)
    dists = np.dot(model.vectors, word_vec)
    ## map cosines to (0,1) range
    ## where higher values indicate higher distance:
    dists = to_dist(dists)

    sorted_dist_ids = matutils.argsort(dists, reverse=True)
    
    if pos_after_word:
        word_distances = [
            (model.index2entity[word_id], float(dists[word_id]))
            for word_id in sorted_dist_ids \
                if model.index2entity[word_id].endswith(word.split('_')[-1])\
                    and float(dists[word_id]) > thresh
        ]
    else:
        raise NotImplementedError

    if max_count:
        word_distances = word_distances[:max_count]
    
    return word_distances
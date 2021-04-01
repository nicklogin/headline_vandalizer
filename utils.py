import lemminflect

def two_way_map(l):
    '''Функция, позволяющая из списка уникальных элементов получить
    два отображения:
    
    индекс-элемент
    элемент-индекс'''
    id2elem = [i for i in l]
    elem2id = {id2elem[i]:i for i in range(len(id2elem))}
    return id2elem, elem2id

def with_reverse_map(func, *args, **kwargs):
    def add_reverse_map(*args, **kwargs):
        func_result = func(*args, **kwargs)
        return two_way_map(func_result)
    return add_reverse_map

def inflect(lemma, xpos):
    inflections = lemminflect.getInflection(lemma, xpos)
    if inflections:
        return inflections[0]
    else:
        return lemma

def last_underscore_split(string):
    a = string[:string.rfind('_')]
    b = string[string.rfind('_')+1:]
    return a, b
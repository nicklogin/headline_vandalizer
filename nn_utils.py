import torch as tt
import re

from tqdm import tqdm as tqdm_notebook
from math import ceil

class UnsupervisedBatch:
  def __init__(self, sents_batch):
    self.masked = sents_batch

class UnsupervisedBatchIterator:
  def __init__(self, sents, batch_size=8):
    self.sents = sents
    self.batch_size = batch_size
  
  def __iter__(self):
    self.start = 0
    return self
  
  def __next__(self):
    if self.start >= len(self.sents):
      raise StopIteration
    batch = self.sents[self.start:self.start+self.batch_size]
    batch = UnsupervisedBatch(batch)
    self.start += self.batch_size
    return batch
  
  def __len__(self):
    return ceil(len(self.sents)/self.batch_size)

def tokenize_batch(batch, tokenizer, max_length=128, unsupervised=False):
  if unsupervised:
    labels = None
    inputs = tokenizer(batch.masked, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
  else:
    labels = tokenizer(batch.fun, return_tensors='pt', padding=True, truncation=True, max_length=max_length)['input_ids']
    inputs = tokenizer(batch.masked, return_tensors='pt', padding='max_length', truncation=True, max_length=labels.size()[1])

  return inputs, labels

def make_predictions(batch_iter, model, tokenizer, verbose=False):
  model.eval()

  predicted_sentences = []
  predicted_words = []

  with tt.no_grad():
    if verbose:
      print('Making predictions...')
      iterator = tqdm_notebook(batch_iter, total=len(batch_iter))
    else:
      iterator = batch_iter
    for batch in iterator:
      inputs, targets = tokenize_batch(batch, tokenizer, unsupervised=True)
      output = model(**inputs)['logits'].argmax(dim=2).detach().numpy()

      for sent in output:
        decoded_sent = tokenizer.decode(sent)
        # remove special tokens:
        decoded_sent = re.sub(r'^\[CLS\]', '', decoded_sent)
        decoded_sent = re.sub(r'\[SEP\]\s*?(\[PAD\]\s*?)*$', '', decoded_sent)
        decoded_sent = decoded_sent.strip()
        predicted_sentences.append(decoded_sent)

      masked_ids = inputs['input_ids'].numpy() == tokenizer.mask_token_id

      for sent_id, sent in enumerate(masked_ids):
        masked_seq = []
        for token_id, token in enumerate(sent):
          if token:
            masked_seq.append(output[sent_id][token_id])
        
        predicted_span = tokenizer.decode(masked_seq)
        predicted_words.append(predicted_span)
  return predicted_sentences, predicted_words
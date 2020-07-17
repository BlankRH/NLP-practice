import string
import torch
import numpy as np
import csv
import os
from torchtext import data, datasets
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pandas as pd
from gensim.models import KeyedVectors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenizer(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split()

def split_file():
    data = pd.read_csv('data\\raw.tsv', sep='\t')
    train, val = train_test_split(data, test_size=0.2)
    train.to_csv("data\\train.csv", index=False)
    val.to_csv("data\\val.csv", index=False)

def get_data(batch_size):
    TEXT = data.Field(sequential=True, tokenize=tokenizer, stop_words=stopwords.words('english'), lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train, val = data.TabularDataset.splits(path='data', train='train.csv',
                                                  validation='val.csv',
                                                  format='csv', skip_header=True,
                                                  fields=[('PhraseId', None), ('SentenceId', None),
                                                          ('Phrase', TEXT), ('Sentiment', LABEL)])
    test = data.TabularDataset('data\\test.tsv', format='tsv', skip_header=True,
                               fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)])

    TEXT.build_vocab(train)
    vocab_size = len(TEXT.vocab)
    train_iter = data.BucketIterator(train, batch_size=batch_size,
                                     sort_key=lambda x: len(x.Phrase),
                                     shuffle=True, device=DEVICE)
    val_iter = data.BucketIterator(val, batch_size=batch_size,
                                     sort_key=lambda x: len(x.Phrase),
                                     shuffle=True, device=DEVICE)
    # BucketIterator自动进行shuffle和bucket
    test_iter = data.Iterator(dataset=test, batch_size=batch_size, train=False, sort=False, device=DEVICE)
    return TEXT.vocab, train_iter, val_iter, test_iter

if __name__ == '__main__':

    #split_file()
    w2v_model = KeyedVectors.load_word2vec_format('data\\GoogleNews-vectors-negative300.bin',
                                                  binary=True)
    print(w2v_model['<UNK>'])
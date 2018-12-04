from keras.models import Sequential
from keras.layers import Input, GRU, Embedding, Bidirectional, LSTM, Dropout, Multiply, Conv1D, Dense, Flatten
import numpy as np
import re
from keras import backend as K
from collections import Counter


max_length = 20

# from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))




vocab = Counter()
train_file = 'train.txt'
test_file = 'test.txt'

with open(train_file, 'r', encoding='utf16') as f:
    for line in f:
        x,y = line.replace('\n','').split('\t')
        for ch in x:
            vocab[ch]+=1

vocab = list(w for w,c in vocab.most_common())
vocab = vocab[:6000]
print('Vocab:',len(vocab),vocab)

def test_unicode_type(ch):
    if u'\u3040' <= ch <= u'\u309F':
        return 0 # Hiragana
    elif u'\u4e00' <= ch <= u'\u9fff':
        return 1 # kanji
    elif 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
        return 2 # romaji
    else:
        return 3 # punctuation


def train_generator(file):
    with open(file, 'r', encoding='utf16') as f:
        for line in f:
            split = line.strip().split('\t')
            if len(split)!=2: continue
            i,o = split
            i = i[:max_length].rjust(max_length)
            x = []
            y = []
            for ch in i:
                x.append(vocab.index(ch) if ch in vocab else len(vocab))
            o = re.sub('.[|]','|', o)
            o = o[:max_length].rjust(max_length)
            for j,ch in enumerate(o):
                y.append(1 if ch == '|' else 0)
            if len(x)!=len(y) or len(x) != max_length:
                continue
            y = np.array([y]).astype('float32')
            x = np.array([x]).astype('float32')
            # print(x.shape)
            # print(y.shape)
            yield x, y


deep_morph = Sequential()
deep_morph.add(Embedding(len(vocab)+1, 256, input_length=max_length))
deep_morph.add(Dropout(0.2))
deep_morph.add(Conv1D(64, kernel_size=(3,), strides=1,
                 activation='relu',
                 padding='same',
                 input_shape=(50,1)))
deep_morph.add(Dropout(0.2))
deep_morph.add(Conv1D(64, kernel_size=(3,), strides=1,
                 activation='relu',
                 padding='same',))
deep_morph.add(Dropout(0.2))
deep_morph.add(Conv1D(1, kernel_size=(3,), strides=1,
                 activation='relu',
                 padding='same',))
deep_morph.add(Dropout(0.2))
deep_morph.add(Flatten())
deep_morph.add(Dense(max_length, activation='sigmoid'))


deep_morph.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy', f1, precision, recall])
print(deep_morph.summary())

# Training
deep_morph.fit_generator(train_generator(train_file), epochs=100, steps_per_epoch=50)

# Evaluation
print(deep_morph.evaluate_generator(train_generator(train_file), steps=128))
print(deep_morph.evaluate_generator(train_generator(test_file), steps=128))



'''
LSTM
units: Positive integer, dimensionality of the output space.
activation: Activation function to use (see activations). 
            Default: hyperbolic tangent (tanh). If you pass None, 
            no activation is applied (ie. "linear" activation: a(x) = x).
recurrent_activation: Activation function to use for the recurrent step (see activations). 
            Default: hard sigmoid (hard_sigmoid). If you pass None, no activation is applied 
            (ie. "linear" activation: a(x) = x).
return_sequences: Boolean. Whether to return the last output in the output sequence,
            or the full sequence.
return_state: Boolean. Whether to return the last state in addition to the output.
go_backwards: Boolean (default False). If True, process the input sequence backwards 
            and return the reversed sequence.
stateful: Boolean (default False). If True, the last state for each sample at index i in a batch 
            will be used as initial state for the sample of index i in the following batch.
unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop 
            will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. 
            Unrolling is only suitable for short sequences.
'''

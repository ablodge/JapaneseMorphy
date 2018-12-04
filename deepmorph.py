from keras.models import Sequential
from keras.layers import Input, GRU, Embedding, Bidirectional, LSTM, Dropout, Multiply
import numpy as np
import re

vocab = set()


train_file = 'train.txt'
test_file = 'test.txt'

with open(train_file, 'r', encoding='utf16') as f:
    for line in f:
        x,y = line.replace('\n','').split('\t')
        for ch in x:
            vocab.add(ch)

vocab = list(vocab)
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
            x = []
            y = []
            for ch in i:
                x.append(vocab.index(ch)+4 if ch in vocab else len(vocab)+4)
                y.append(test_unicode_type(ch))
            o = re.sub('.[|]','|', o)
            # for j,ch in enumerate(o):
            #     y.append(1 if ch == '|' else 0)
            #     y.append(1 if ch == '|' else 0)
            if len(x)!=len(y): continue
            y = np.array(y).astype('float32')
            x = np.array(x).astype('float32')
            yield x, y

# chars = Sequential()
# features = Sequential()
# chars = Sequential()
# chars.add(GRU(32, input_shape=(None, 1)))
# feats = Sequential()
# feats.add(GRU(32, input_shape=(None, 1)))


deep_morph = Sequential()
# deep_morph.add(Merge([chars, feats]))
deep_morph.add(Embedding(len(vocab)+5, 256))
deep_morph.add(Dropout(0.2))
# deep_morph.add(GRU(16,return_sequences=True))
deep_morph.add(Bidirectional(GRU(64,activation='relu',return_sequences=True)))
deep_morph.add(Dropout(0.2))
deep_morph.add(Bidirectional(GRU(64,activation='relu',return_sequences=True)))
deep_morph.add(Dropout(0.2))
deep_morph.add(GRU(1,activation='sigmoid'))
deep_morph.add(Dropout(0.2))


deep_morph.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
print(deep_morph.summary())

# Training
deep_morph.fit_generator(train_generator(train_file), epochs=100, steps_per_epoch=30)

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

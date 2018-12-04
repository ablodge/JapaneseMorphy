from keras.models import Sequential
from keras.layers import Input, GRU, Embedding, Bidirectional, LSTM, Dropout, Multiply
import numpy as np
import re
from keras import backend as K

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
        return [1,0,0,0] # Hiragana
    elif u'\u4e00' <= ch <= u'\u9fff':
        return [0,1,0,0] # kanji
    elif 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
        return [0,0,1,0] # romaji
    else:
        return [0,0,0,1] # punctuation


def train_generator_word_level(file):
    with open(file, 'r', encoding='utf16') as f:
        for line in f:
            split = line.strip().split('\t')
            if len(split)!=2: continue
            i,o = split

            x = [vocab.index(ch) if ch in vocab else len(vocab) for ch in i]
            y = [test_unicode_type(ch) for ch in i]
            y = np.array(y).astype('float32')
            x = np.array(x).astype('float32')
            yield x,y


def train_generator(file):
    with open(file, 'r', encoding='utf16') as f:
        for line in f:
            split = line.strip().split('\t')
            if len(split)!=2: continue
            i,o = split
            x = []
            y = []
            for ch in i:
                x.append(vocab.index(ch) if ch in vocab else len(vocab))
            o = re.sub('.[|]','|', o)
            for j,ch in enumerate(o):
                y.append(1 if ch == '|' else 0)
            if len(x)!=len(y):
                continue
            y = np.array(y).astype('float32')
            x = np.array(x).astype('float32')
            yield x, y

# chars = Sequential()
# features = Sequential()
# chars = Sequential()
# chars.add(GRU(32, input_shape=(None, 1)))
# feats = Sequential()
# feats.add(GRU(32, input_shape=(None, 1)))

emb = Embedding(len(vocab)+1, 256)



deep_morph = Sequential()
# deep_morph.add(Merge([chars, feats]))
deep_morph.add(emb)
deep_morph.add(Dropout(0.2))
deep_morph.add(GRU(4,activation='softmax'))
deep_morph.add(Dropout(0.2))

from keras import optimizers
deep_morph.compile(optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.0)
                   , loss='categorical_crossentropy', metrics=['accuracy'])
print(deep_morph.summary())

# Training
deep_morph.fit_generator(train_generator_word_level(train_file), epochs=100, steps_per_epoch=100)

# Evaluation
print(deep_morph.evaluate_generator(train_generator_word_level(train_file), steps=128))
print(deep_morph.evaluate_generator(train_generator_word_level(test_file), steps=128))




deep_morph = Sequential()
# deep_morph.add(Merge([chars, feats]))
deep_morph.add(emb)
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
deep_morph.fit_generator(train_generator(train_file), epochs=100, steps_per_epoch=100)

# Evaluation
print(deep_morph.evaluate_generator(train_generator(train_file), steps=128))
print(deep_morph.evaluate_generator(train_generator(test_file), steps=128))



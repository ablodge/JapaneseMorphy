import torch
from torch import nn as nn
import re
from collections import Counter

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



def train_generator_word_level(file):
    with open(file, 'r', encoding='utf16') as f:
        for line in f:
            split = line.strip().split('\t')
            if len(split)!=2: continue
            i,o = split
            o = o.split('|')
            for tok in o:
                if not tok: continue
                x = [vocab.index(ch) if ch in vocab else len(vocab) for ch in tok]
                y = [0 for ch in tok]
                y[-1] = 1
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
            yield x, y


def accuracy(y_pred, y_true):
    assert len(y_pred)==len(y_true), 'Warning in Accuracy'
    acc = 0
    for y1,y2 in zip(y_pred, y_true):
        y1 = 1 if y1 >= 0.5 else 0
        acc += 1 if y1 == y2 else 0

    acc/=float(len(y_true))
    return acc

def train(data_generator, gru, loss_f, opt, epochs=10, batches=1):
    for epoch in range(epochs):  # loop over the dataset multiple times
        x_batch = []
        y_batch = []
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(data_generator):
            inputs, labels = data
            x_batch.append(inputs)
            y_batch.append(labels)

            if len(x_batch) == batches:

                # zero the parameter gradients
                opt.zero_grad()

                x_batch = torch.tensor(x_batch, dtype=torch.long).cuda()
                y_batch = torch.tensor(y_batch, dtype=torch.float32).cuda()

                outputs, hn = gru(x_batch)
                # print(outputs.size())
                outputs = nn.Sigmoid()(outputs)
                # print(outputs.size())
                outputs = outputs.view(1,-1)
                loss = loss_f(outputs, y_batch)
                loss.backward()
                opt.step()

                # print statistics
                running_loss += loss.item()
                for y_pred, y_true in zip(outputs, y_batch):
                    running_accuracy += accuracy(y_pred,y_true)
                if i%1000 == 0 and i>0:
                    print(f'epoch {epoch+1} item {i} loss: {running_loss/2000} acc {running_accuracy/2000}')
                    running_loss = 0.0
                    running_accuracy = 0.0
                x_batch = []
                y_batch = []
    print('Finished Training')


gru = nn.Sequential(
    nn.Embedding(len(vocab) + 1, 256),
    nn.ReLU(),
    nn.GRU(256, 1, num_layers=3, bidirectional=False, batch_first=True),
)
data = train_generator(train_file)
loss_f = nn.BCELoss()
opt = torch.optim.Adadelta(gru.parameters())
print('Loaded Data')
train(data, gru, loss_f, opt, epochs=10)






'''
GRU
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
num_layers – Number of recurrent layers. E.g., setting num_layers=2 
            would mean stacking two GRUs together to form a stacked GRU, 
            with the second GRU taking in outputs of the first GRU and 
            computing the final results. 
            Default: 1
bias – If False, then the layer does not use bias weights b_ih and b_hh. 
            Default: True
batch_first – If True, then the input and output tensors are provided as 
            (batch, seq, feature). 
            Default: False
dropout – If non-zero, introduces a Dropout layer on the outputs of each GRU 
            layer except the last layer, with dropout probability equal to dropout. 
            Default: 0
bidirectional – If True, becomes a bidirectional GRU. 
            Default: False
            
Input (seq_len, batch, input_size): tensor containing the features of the input sequence. 
            The input can also be a packed variable length sequence. See 
            torch.nn.utils.rnn.pack_padded_sequence() for details.
Output h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the 
            initial hidden state for each element in the batch. Defaults to zero if not provided.
'''

'''
Embeddings
num_embeddings (int) – size of the dictionary of embeddings
embedding_dim (int) – the size of each embedding vector
padding_idx (int, optional) – If given, pads the output with 
            the embedding vector at padding_idx (initialized to zeros) 
            whenever it encounters the index.
max_norm (float, optional) – If given, will renormalize the embedding 
            vectors to have a norm lesser than this before extracting.
norm_type (float, optional) – The p of the p-norm to compute for the 
            max_norm option. Default 2.
scale_grad_by_freq (boolean, optional) – if given, this will scale gradients 
            by the inverse of frequency of the words in the mini-batch. Default False.
sparse (bool, optional) – if True, gradient w.r.t. weight matrix will be a sparse 
            tensor. See Notes for more details regarding sparse gradients.
            
Input: LongTensor of arbitrary shape containing the indices to extract
Output: (*, embedding_dim), where * is the input shape
'''

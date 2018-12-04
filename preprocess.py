import os, random

input_dir = './data'
train_file = 'train.txt'
test_file = 'test.txt'
X = {}
for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        input_file = os.path.join(input_dir,file)
        with open(input_file, 'r', encoding='utf16') as f:
            for i, line in enumerate(f):
                if i == 0: continue
                split = line.strip().split('\t')
                y = split[3] + split[4] + split[5]
                ys = y.split('#')
                xs = [y.replace('|', '') for y in ys]
                for x, y in zip(xs, ys):
                    X[x.strip()] = y.strip()

Y = [X[x] for x in X]
X = [x for x in X]
indices = [i for i,x in enumerate(X)]
random.shuffle(indices)

i = 0
size = float(len(Y))
with open(test_file, 'w+', encoding='utf16') as f2:
    with open(train_file, 'w+', encoding='utf16') as f1:
        for j in indices:
            if i / size < 0.8:
                f1.write(X[j] + '\t' + Y[j] + '\n')
                print(X[j],Y[j])
            else:
                f2.write(X[j] + '\t' + Y[j] + '\n')
            i += 1

print('Sentences:',int(size))
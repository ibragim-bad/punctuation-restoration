import string

punc_set = {k:v for v,k in enumerate(string.punctuation)}

PNC = {',':'PERIOD','.':'COMMA',
'QUESTION':'?'}
pnc = ',.?'
punc_set = {k:v for v,k in zip(list(range(1,len(pnc))), pnc)}
extra_punc = list(set(string.punctuation) - set(pnc))
sps = [' ' + p for p in pnc]

import re, string
pattern = re.compile('[^a-zA-ZА-Яа-я0-9,?.\s]+')


def clean_extra(s):
    s = s.replace('!', '.')
    s = s.replace('...', '.')
    s = s.replace(':', ',')
    s = s.replace(';', ',')
    # for i in extra_punc:
    #     s = s.replace(i, '')
    s = pattern.sub('', s)
    l = s.split()
    for i,w in enumerate(l):
      if l[i] in pnc:
        p = l.pop(i)
        l[i-1] = l[i-1] + p
    return " ".join(l)

def get_targets(s):
    ids = []
    for w in s:
        p = PNC.get(w[-1], 'O') 
        ids.append(p)
    return ids

def clean_targets(sl):
    al = []
    for s in sl:
        s = s.lower()
        for i in pnc:
            s = s.replace(i, '')
        al.append(s)
    return al

import pandas as pd

FN = '/Users/ibragim/Documents/ytsubs/ru-ted.txt'
with open(FN) as f: txt = f.read()

txt = clean_extra(txt)
tl = txt.split()
y = get_targets(tl)
x = clean_targets(tl)


df = pd.DataFrame({'text': x, 'label': y})


from sklearn.model_selection import train_test_split

coms = df[df.label == 'COMMA']
splitter = coms.index.values[len(coms) - len(coms)//8]

train, test = df.iloc[:splitter + 1], df.iloc[splitter + 1:]

train.to_csv('data/train_ru.tsv', sep='\t',index=False, header=False)
test.to_csv('data/test_ru.tsv', sep='\t', index=False, header=False)
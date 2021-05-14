import re, string
import pandas as pd
from sklearn.model_selection import train_test_split

FN = '/Users/ibragim/Documents/ytsubs/ru-ted.txt'

class Preprocessor(object):

    def __init__(self, cap=True):
        self.pattern = re.compile('[^a-zA-ZА-Яа-я0-9,?.-:;★\s]+')
        self.punc_set = {k:v for v,k in enumerate(string.punctuation)}
        self.pnc = {'0':'O',',':'PERIOD','.':'COMMA', '?':'QUESTION', ':':'COLON', ';':'SEMICOLON', '-':'DASH',
                   '★':'SLASH'}
        self.cap = True

    def clean_extra(self, s):
        s = s.replace('ё', 'е')
        s = s.replace('\n', '★ ')
        s = s.replace('!', '.')
        s = s.replace('...', '.')
        s = self.pattern.sub('', s)
        l = s.split()
        for i,w in enumerate(l):
            if l[i] in self.pnc:
                p = l.pop(i)
                l[i-1] = l[i-1] + p
        return " ".join(l)
    

    def get_targets(self, s):
        ids = []
        for w in s:
            p = self.pnc.get(w[-1], 'O') 
            ids.append(p)
        return ids

    def is_capit(self, s):
        ids = []
        for w in s:
            p = 1 if w[0].isupper() else 0 
            ids.append(p)
        return ids
        
    def get_targets(self, s):
        ids = []
        for w in s:
            c = ''
            p = self.pnc.get(w[-1], 'O')
            if self.cap:
                c = '1' if w[0].isupper() else '0' 
            p = ''.join([c,p])
            ids.append(p)
        return ids

    def clean_targets(self, sl):
        al = []
        for s in sl:
            s = s.lower()
            for i in self.pnc:
                s = s.replace(i, '')
            al.append(s)
        return al
    
    def prep_txt(self,txt):
        txt = self.clean_extra(txt)
        tl = txt.split()
        y = self.get_targets(tl)
        x = self.clean_targets(tl)
        df = pd.DataFrame({'text': x, 'label': y})
        return df

    def prep_file(self,fn):
        with open(fn) as f: 
            txt = f.read()
        df = self.prep_txt(txt)
        return df

if __name__ == '__main__':
    import sys
    fn = sys.argv[1]
    tof = sys.argv[2]
    prp = Preprocessor()
    t = prp.prep_file(fn)
    t.to_csv(t, sep='\t',index=False, header=False)




# coms = df[df.label == 'COMMA']
# splitter = coms.index.values[len(coms) - len(coms)//8]

# train, test = df.iloc[:splitter + 1], df.iloc[splitter + 1:]

# train.to_csv('data/train_ru.tsv', sep='\t',index=False, header=False)
# test.to_csv('data/test_ru.tsv', sep='\t', index=False, header=False)
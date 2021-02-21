from typing import List
import youtokentome as yttm
from itertools import chain

class YTTM:

    def __init__(self, path):
        self.model = yttm.BPE(path, n_threads=-1)

    def tokenize(self, word: str):
        ix = self.model.encode(word, output_type=yttm.OutputType.SUBWORD)
        return ix

    def convert_tokens_to_ids(self, text_l, imit=False):
        text_l = self.model.subword_to_id(text_l) 
        return text_l
    
    @property
    def vocab_size(self):
        return self.model.vocab_size()
import re
import torch

from datetime import datetime

from yttm import YTTM
from lstm_model import BiLSTM_CNN_CRF


YTTM_PATH = 'yttm.model'
DEV = 'cpu'
MODEL_PATH = 'out/weights.pt'
SEQ_LEN = 92

tokenizer = YTTM(YTTM_PATH)
deep_punctuation = BiLSTM_CNN_CRF(4, tokenizer.vocab_size, 512)
deep_punctuation.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEV)))
deep_punctuation.eval()


def inference(text):
    text = re.sub(r"[,:\-–.!;?]", '', text)
    print(len(text))
    print(len(text.split()))

    bos = '<BOS>'
    eos = '<EOS>'
    pad = '<PAD>'
    unk = '<UNK>'

    words_original_case = text.split()
    words = text.lower().split()
    words = words

    word_pos = 0
    sequence_len = SEQ_LEN
    result = ""
    decode_idx = 0
    punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}

    while word_pos < len(words):
        x = [bos]
        y_mask = [0]

        a = datetime.now()
        while len(x) < sequence_len and word_pos < len(words):
            tokens = tokenizer.tokenize(words[word_pos])
            if len(tokens) + len(x) >= sequence_len:
                break
            else:
                for i in range(len(tokens) - 1):
                    x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                    y_mask.append(0)
                x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                y_mask.append(1)
                word_pos += 1
        x.append(eos)
        y_mask.append(0)
        if len(x) < sequence_len:
            x = x + [pad for _ in range(sequence_len - len(x))]
            y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
        attn_mask = [1 if token != pad else 0 for token in x]

        x = torch.tensor(x)
        y_mask = torch.tensor(y_mask)
        attn_mask = torch.tensor(attn_mask)
        x, attn_mask, y_mask = x.to(DEV), attn_mask.to(DEV), y_mask.to(DEV)
        b = datetime.now()
        print('tokenize')
        print(b - a)
        
        with torch.no_grad():
            c = datetime.now()
            y_predict = deep_punctuation(x, attn_mask)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            y_predict = torch.argmax(y_predict, dim=1).view(-1)
            d = datetime.now()
            print(f'words len {len(words)}')
            print('model')
            print(d - c)
        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
                decode_idx += 1
        return result



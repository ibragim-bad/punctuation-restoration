import torch.nn as nn
import torch
from config import *
from torchcrf import CRF

from transformers import BertModel, BertConfig

class Student(nn.Module):
    def __init__(self, config, freeze_bert=False, lstm_dim=-1):
        super(Student, self).__init__()
        self.output_dim = len(punctuation_dict)
        configuration = BertConfig(**config, output_hidden_states=True)
        self.bert_layer = BertModel(configuration)
        # Freeze bert layers
        # if freeze_bert:
        #     for p in self.bert_layer.parameters():
        #         p.requires_grad = False
        bert_dim = config['hidden_size']
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=len(punctuation_dict))
        self.proj_lin_1 = nn.Linear(in_features=hidden_size, out_features=768)
        self.proj_lin_2 = nn.Linear(in_features=hidden_size, out_features=768)
        self.proj_lin_3 = nn.Linear(in_features=hidden_size, out_features=768)


    def forward(self, input_ids, attention_mask):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.view(1, input_ids.shape[0])
            attention_mask = attention_mask.view(1, -1)  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        out = self.bert_layer(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        x = out.last_hidden_state
        hs = out.hidden_states
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)

        pr1 = self.proj_lin_1(hs[0])
        pr2 = self.proj_lin_2(hs[1])
        pr3 = self.proj_lin_3(hs[2])
        return x, pr1, pr2, pr3

class Teacher(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=True, lstm_dim=-1):
        super(Teacher, self).__init__()
        self.output_dim = len(punctuation_dict)
        self.config = BertConfig.from_pretrained(pretrained_model,)
        self.bert_layer = BertModel(self.config)
        # Freeze bert layers
        # if freeze_bert:
        for p in self.bert_layer.parameters():
            p.requires_grad = False
        bert_dim = self.config.hidden_size
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)


    def forward(self, input_ids, attention_mask):
        # if len(x.shape) == 1:
        #     x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        out = self.bert_layer(input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)

        return x, hs[0], hs[6], hs[12]



from transformers import BertConfig

class DeepPunctuation(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuation, self).__init__()
        self.output_dim = len(punctuation_dict)
        config = BertConfig.from_pretrained(pretrained_model)
        bert_cl =  MODELS[pretrained_model][0]
        #self.bert_layer = bert_cl(config)
        self.bert_layer = bert_cl.from_pretrained(pretrained_model)
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = MODELS[pretrained_model][2]
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=len(punctuation_dict))

    def forward(self, x, attn_masks, distil=False):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])
            attn_masks = attn_masks.view(1, -1)
            #print(x.shape, attn_masks.shape)  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        out = self.bert_layer(x, attention_mask=attn_masks)
        x = out.last_hidden_state
        hs = 0
        if distil:
          hs = out.hidden_states
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x


class DeepPunctuationCRF(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuationCRF, self).__init__()
        self.bert_lstm = DeepPunctuation(pretrained_model, freeze_bert, lstm_dim)
        self.crf = CRF(len(punctuation_dict), batch_first=True)

    def log_likelihood(self, x, attn_masks, y):
        x = self.bert_lstm(x, attn_masks)
        attn_masks = attn_masks.byte()
        return -self.crf(x, y, mask=attn_masks, reduction='token_mean')

    def forward(self, x, attn_masks, y):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        x = self.bert_lstm(x, attn_masks)
        attn_masks = attn_masks.byte()
        dec_out = self.crf.decode(x, mask=attn_masks)
        y_pred = torch.zeros(y.shape).long().to(y.device)
        for i in range(len(dec_out)):
            y_pred[i, :len(dec_out[i])] = torch.tensor(dec_out[i]).to(y.device)
        return y_pred

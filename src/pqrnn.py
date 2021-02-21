
import math
import warnings
from functools import partial
from typing import Any, Dict, List
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import LSTM as QRNN
from murhash import murmurhash

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PQRNN(nn.Module):
    def __init__(
        self,
        b: int = 256,
        d: int = 64,
        num_layers: int = 2,
        fc_sizes: List[int] = None,
        output_size: int = 4,
        lr: float = 0.025,
        dropout: float = 0.5,
        rnn_type: str = "GRU",
        multilabel: bool = False,
        nhead: int = 8,
    ):
        super().__init__()
        if fc_sizes is None:
            fc_sizes = [128, 64]

        self.hparams: Dict[str, Any] = {
            "b": b,
            "d": d,
            "fc_size": fc_sizes,
            "lr": lr,
            "output_size": output_size,
            "dropout": dropout,
            "rnn_type": rnn_type.upper(),
            "multilabel": multilabel,
            "nhead": nhead,
        }

        layers: List[nn.Module] = []
        for x, y in zip([d] + fc_sizes, fc_sizes + [output_size]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(x, y))

        self.tanh = nn.Hardtanh()
        if self.hparams["rnn_type"] in {"LSTM", "GRU"}:
            self.hidden = {
                "LSTM": partial(nn.LSTM, bidirectional=True),
                "GRU": partial(nn.GRU, bidirectional=True),
            }[self.hparams["rnn_type"]](
                b, d, num_layers=num_layers, dropout=dropout
            )
        else:
            self.pos_encoder = PositionalEncoding(d_model=b, dropout=dropout)
            encoder_layers = TransformerEncoderLayer(
                d_model=b, nhead=nhead, dropout=dropout
            )
            self.hidden = TransformerEncoder(
                encoder_layers, num_layers=num_layers
            )
            self.linear = nn.Linear(b, d)

        self.output = nn.ModuleList(layers)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, projection, attention_mask):
        features = self.tanh(projection)
        features = features.transpose(0, 1)
        if self.hparams["rnn_type"] in {"LSTM", "GRU", "QRNN"}:
            output, _ = self.hidden(features)
            if self.hparams["rnn_type"] != "QRNN":
                output = (
                    output[..., : output.shape[-1] // 2]
                    + output[..., output.shape[-1] // 2 :]
                )
        else:
            features = features * math.sqrt(self.hparams["b"])
            features = self.pos_encoder(features)
            output = self.hidden(
                features,
                self.generate_square_subsequent_mask(features.size(0)).to(
                    features.device
                ),
            )
            output = self.linear(output)
        output = output.transpose(0, 1)
        logits = output#torch.mean(output, dim=1)
        for layer in self.output:
            logits = layer(logits)
        return logits

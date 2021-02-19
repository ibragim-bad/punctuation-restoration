import torch
import copy
import torch.nn as nn
from argparser import parse_arguments
from dataset import Dataset
from model import DeepPunctuation, DeepPunctuationCRF
from config import *
import augmentation
import os

torch.multiprocessing.set_sharing_strategy('file_system') 

args = parse_arguments()


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.bert_layer.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, len(num_layers_to_keep)):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.bert_layer.encoder.layer = newModuleList

    return copyOfModel

if __name__ == '__main__':

    device = torch.device('cpu')
    deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm_dim=-1)
    model_save_path = os.path.join(args.save_path, 'weights.pt')
    new_model_save_path = os.path.join(args.save_path, 'pruned_weights.pt')

    deep_punctuation.to(device)
    deep_punctuation.load_state_dict(torch.load(model_save_path, map_location=torch.device(device)))
    # seq = torch.nn.Sequential(*(list(deep_punctuation.children()))) 
    # print(len(seq))
    # # print(deep_punctuation)
    # bert_module = seq[0]

    # seq_bert = torch.nn.Sequential(*(list(bert_module.encoder.children()))) 
    # print(len(bert_module))
    ls = [0,6,12]
    new_model = deleteEncodingLayers(deep_punctuation, ls)
    # nm = list(new_model.children()) + seq[:1]
    # new_dp = torch.nn.Sequential(*nm) 
    print(new_model)
    torch.save(new_model.state_dict(), new_model_save_path)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(deep_punctuation.parameters(), lr=args.lr, weight_decay=args.decay)
#deep_punctuation.load_state_dict(torch.load(model_save_path, map_location=torch.device(device)))

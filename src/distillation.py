import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import torch.multiprocessing
from tqdm import tqdm
import random
from glob import glob
from prep import Preprocessor
import pandas as pd

from argparser import parse_arguments
from dataset import Dataset
from model import DeepPunctuation, Student
from config import *
import augmentation
import json

torch.multiprocessing.set_sharing_strategy('file_system') 

args = parse_arguments()

# for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# tokenizer
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
augmentation.tokenizer = tokenizer
augmentation.sub_style = args.sub_style
augmentation.alpha_sub = args.alpha_sub
augmentation.alpha_del = args.alpha_del
token_style = MODELS[args.pretrained_model][3]
ar = args.augment_rate
sequence_len = args.sequence_length
aug_type = args.augment_type

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 1
}

# logs
os.makedirs(args.save_path, exist_ok=True)
model_save_path = os.path.join(args.save_path, 'weights.pt')
st_model_save_path = os.path.join(args.save_path, 'st_weights.pt')
log_path = os.path.join(args.save_path, args.name + '_logs.txt')


# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')

deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)
criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()
# optimizer = torch.optim.Adam(deep_punctuation.parameters(), lr=args.lr, weight_decay=args.decay)
deep_punctuation.load_state_dict(torch.load(model_save_path, map_location=torch.device(device)))

with open(args.student) as f:
    cfg = json.load(f)

studet_deep = Student(cfg)
studet_deep.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(studet_deep.parameters(), lr=args.lr, weight_decay=args.decay)

def validate(data_loader):
    """
    :return: validation accuracy, validation loss
    """
    num_iteration = 0
    studet_deep.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='eval'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict, _,_,_ = studet_deep(x, att, y)
                loss = studet_deep.log_likelihood(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict, _,_,_ = studet_deep(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                loss = criterion(y_predict, y)
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            val_loss += loss.item()
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
    return correct/total, val_loss/num_iteration


def test(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    num_iteration = 0
    studet_deep.eval()
    # +1 for overall result
    tp = np.zeros(1+len(punctuation_dict), dtype=np.int)
    fp = np.zeros(1+len(punctuation_dict), dtype=np.int)
    fn = np.zeros(1+len(punctuation_dict), dtype=np.int)
    cm = np.zeros((len(punctuation_dict), len(punctuation_dict)), dtype=np.int)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='test'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict, _,_,_ = studet_deep(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict, _,_,_ = studet_deep(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be any punctuation in this position
                    # since we created this position due to padding or sub-word tokenization
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1
    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, correct/total, cm


def train():
    with open(log_path, 'a') as f:
        f.write(str(args)+'\n')
    best_val_acc = 0
    prepr = Preprocessor()

    fs = [f for f in glob(args.data_path)]
    
    for epoch in range(args.epoch):
        fn = random.choice(fs)
 
        try:
            df = prepr.prep_file(fn)
        except:
            print(f'error {fn}')
            fn = random.choice(fs)
            df = prepr.prep_file(fn)
        str_df = df.to_csv(sep='\t', header=None, index=False)
        train_set = Dataset(str_df, tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
        
        train_loader = torch.utils.data.DataLoader(train_set, **data_loader_params)

        train_loss = 0.0
        mse_loss = 0.0
        train_iteration = 0
        correct = 0
        total = 0
        studet_deep.train()
        for x, y, att, y_mask in tqdm(train_loader, desc='train'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            # if args.use_crf:
            #     loss = deep_punctuation.log_likelihood(x, att, y)
            #     # y_predict = deep_punctuation(x, att, y)
            #     # y_predict = y_predict.view(-1)
            #     y = y.view(-1)
            # else:
            deep_punctuation.eval()  # manually set DeepPunctuation model to eval mode
            with torch.no_grad():
                y_predict_t, hs = deep_punctuation(x, att)

            y_predict, s_hs_1, s_hs_2, s_hs_3 = studet_deep(x, att)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            y = y.view(-1)
            loss = criterion(y_predict, y)

            loss_1 = mse(s_hs_1, hs[0])
            loss_2 = mse(s_hs_2, hs[10])
            #loss_3 = mse(s_hs_3, hs[2])

            wloss =  (loss_1 + loss_2) / 3
            total_loss = (wloss * 0.3 + loss * 0.7) 
            y_predict = torch.argmax(y_predict, dim=1).view(-1)

            correct += torch.sum(y_mask * (y_predict == y).long()).item()

            optimizer.zero_grad()
            train_loss += loss.item()
            mse_loss += wloss.item()
            train_iteration += 1
            total_loss.backward()

            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(studet_deep.parameters(), args.gradient_clip)
            optimizer.step()

            y_mask = y_mask.view(-1)

            total += torch.sum(y_mask).item()

        train_loss /= train_iteration
        mse_loss /= train_iteration
        log = 'epoch: {}, Train loss: {}, MSEL {}, Train accuracy: {}'.format(epoch, train_loss, mse_loss, correct / total)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)

        if epoch % 25 == 0:
            #fn = random.choice(fs)
            df = pd.read_csv('data/test_ru.tsv', sep='\t', header=None)

            #df = prepr.prep_file(fn)
            str_df =df.to_csv(sep='\t', header=None, index=False)
            
            val_set = Dataset(str_df, tokenizer=tokenizer, sequence_len=sequence_len,
                            token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
            
            val_loader = torch.utils.data.DataLoader(val_set, **data_loader_params)
            val_acc, val_loss = validate(val_loader)
            log = 'epoch: {}, Val loss: {}, Val accuracy: {}'.format(epoch, val_loss, val_acc)
            with open(log_path, 'a') as f:
                f.write(log + '\n')
            print(log)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(studet_deep.state_dict(), st_model_save_path)

    # print('Best validation Acc:', best_val_acc)
    # deep_punctuation.load_state_dict(torch.load(model_save_path))
    # for loader in test_loaders:
    #     precision, recall, f1, accuracy, cm = test(loader)
    #     log = 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + 'F1 score: ' + str(f1) + \
    #           '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix' + str(cm) + '\n'
    #     print(log)
    #     with open(log_path, 'a') as f:
    #         f.write(log)
    #     log_text = ''
    #     for i in range(1, 5):
    #         log_text += str(precision[i] * 100) + ' ' + str(recall[i] * 100) + ' ' + str(f1[i] * 100) + ' '
    #     with open(log_path, 'a') as f:
    #         f.write(log_text[:-1] + '\n\n')


if __name__ == '__main__':
    train()

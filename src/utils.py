from __future__ import absolute_import
import os
import argparse
import tqdm
import csv
import numpy as np
import sys
import torch
import math
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from os import listdir
from os.path import isfile, join, exists
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


#load and shuffle data for training
def data_loader(args, train, dev, test):

    #train, dev, test = load_tokenized_data(args.data_path)

    train = pad_batch(train,args.batch_size)
    dev = pad_batch(dev,args.batch_size)
        
    logger.info("%s train samples", len(train))
    logger.info("%s dev samples", len(dev))
    logger.info("%s test samples", len(test))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")

    train_x = [sub['review'] for sub in train ]
    train_x = tokenizer(train_x, padding=True, truncation=True, return_tensors="pt")['input_ids']
    train_y = np.asarray([sub['rating'] for sub in train ])
    dev_x = [sub['review'] for sub in dev]
    dev_x = tokenizer(dev_x, padding=True, truncation=True, return_tensors="pt")['input_ids']
    dev_y = np.asarray([sub['rating'] for sub in dev ])
    test = [sub['review'] for sub in test ]
    test = tokenizer(test, padding=True, truncation=True, return_tensors="pt")['input_ids']

    # dataloaders
    logger.info("Data loader running")
    train_data = TensorDataset(torch.LongTensor(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.LongTensor(dev_x), torch.from_numpy(dev_y))
    test_data = TensorDataset(torch.LongTensor(test))

    #shuffling data
    batch_size = args.batch_size
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)  

    return train_loader, valid_loader, test_loader

def sample_(loader):
    # obtain one batch of data
    dataiter = iter(loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size()) # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size()) # batch_size
    print('Sample label: \n', sample_y)  


def raw_data_loader(data_path):
    logger.info('Loading datasets')
    data_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    data={}
    for file_ in data_files:
        file_path = join(data_path,file_)
        logger.info('Loading dataset from %s', file_path)
        if '_train' in file_:
            data['train'] = csv_reader(file_path,'train')

        elif '_dev' in file_:
            data['dev'] = csv_reader(file_path,'test')

        elif '_test' in file_:
            data['test'] = csv_reader(file_path,'inference')
        else:
            data['inference'] = csv_reader(file_path, 'test')
    return data
        

def csv_reader(fileName,description):
    data =[]
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                keys = row
                line_count += 1
            else:
                if description == 'inference':
                    try:
                        data.append({'id': row[0], 'review':row[1]})
                    except Exception as e:
                        print(e)
                        continue
                else:
                    try:
                        if int(row[2]) < 0 or int(row[2]) > 5:
                            continue
                        else:
                            data.append({'id': row[0], 'review':row[1], 'rating': int(row[2])-1})
                    except Exception as e: 
                        print(e)
                        continue

                line_count += 1
        logger.info(f'{line_count} lines in {fileName}.')
    return data
    
def pad_batch(data,batch_size):
    last_batch_size = (math.ceil(len(data) / batch_size) * batch_size) - len(data)
    if last_batch_size > 0:
        padded_data = data[0:last_batch_size]
        data.extend(padded_data)
    return data

def write_to_csv(out_path, reviews, pred,labels=None):
    with open(out_path, mode='w') as csv_file:
        rev_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not labels is None:
            rev_writer.writerow(['id', 'review', 'rating', 'ref_rating'])
        else:
            rev_writer.writerow(['id','review', 'rating'])

        for rev in tqdm.tqdm(range(len(reviews))):
            if not labels is None:
                rev_writer.writerow([rev, reviews[rev], pred[rev]+1, labels[rev]+1])
            else:
                rev_writer.writerow([rev, reviews[rev], pred[rev]+1])
    logger.info("predictions written to %s", out_path)

def evaluate(reference, pred, description):   
    s= "\n\n" + description + " EVALUATION\n"
    acc = [1 for i in range(len(pred)) if pred[i]==reference[i]]
    acc= sum(acc)/len(pred)
    #print("Accuracy  = " , accuracy(reference,pred))
    s += "Accuracy  = " + str(acc)
    p =  precision_score(reference, pred,average="weighted")
    r =  recall_score(reference, pred, average="weighted")
    if (p+r) == 0:
        f = 0
    else:
        f = 2 * (p * r) / (p + r)
    
    s += "\nPrecision  = " + str(p)
    s += "\nRecall  = " + str(r)
    s += "\nF_measure  = " + str(f)

    logger.info(s)

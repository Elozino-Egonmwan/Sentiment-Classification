import torch.nn as nn
import torch
import numpy as np
import sys
import tqdm
import pickle
import os
import shutil
from os import listdir
from os.path import join, isfile
from torch.utils.data import DataLoader, TensorDataset
from utils import raw_data_loader, data_loader, logger, csv_reader, write_to_csv, evaluate

train_on_gpu = torch.cuda.is_available()

from transformers import AutoModelForSequenceClassification, AutoTokenizer

class FinetuneTransformer(nn.Module):
    def __init__(self, output_size, hidden_dim, n_layers, drop_prob=0.0):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        pretrained_transformer = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # linear and softmax layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        #pretrained classifier
        self.finetune_from = pretrained_transformer
        
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        logits_from_transformer = self.finetune_from(x).logits
        
        out = self.fc(logits_from_transformer)
        
        # softmax function
        softmax_out = self.softmax(out)

        # return last sigmoid output and hidden state
        return softmax_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
            
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

def finetune_mrpc(args):
    
    model = init_trainer(args)

    #if trained model exists
    model_files = [f for f in listdir(args.trained_model_dir) if isfile(join(args.trained_model_dir, f))]
    if model_files:
        MODEL_PATH = join(args.trained_model_dir,model_files[0])
        model.load_state_dict(torch.load(MODEL_PATH))
        logger.info("Trained model loaded from %s", MODEL_PATH)

    if args.mode == "train":
        trainer(args,model)
    elif args.mode == "test":
        validate(args,model)
    else:
        infer(args,model)

def trainer(args,model,preloaded=None):
    #load data
    def load_data():
        data = raw_data_loader(args.data_path)
        train = data['train']
        validate = data['dev']
        test = data['test']

        return train, validate, test

    if not preloaded:
        train, validate, test = load_data()
    else:
        train, validate, test = preloaded

    train_loader, valid_loader, test_loader = data_loader(args, train, validate, test)
    # training params
    epochs = args.epochs
    lr=args.learning_rate
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    clip=args.gradient_clipping

    # move model to GPU, if available
    if(train_on_gpu):
        model.cuda()

    model.train()
    print("Training")
    count = 0
    eval_every = 500
    for e in tqdm.tqdm(range(epochs)):        
        # initialize hidden state
        h = model.init_hidden(args.batch_size)

        train_accs= []
        train_loss=[]
        # batch loop
        for inputs, labels in train_loader:

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()


            # Creating new variables for the hidden state
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            if(train_on_gpu):
                inputs = inputs.type(torch.cuda.LongTensor)
            else:
                inputs = inputs.type(torch.LongTensor)

            output, h = model(inputs, h)
            
            # calculate the loss and perform backprop
            #loss = criterion(output.squeeze(), labels.float())
            
            pred = torch.argmax(output, dim=1)
            acc = torch.sum(pred == labels)/args.batch_size
            train_accs.append(acc.item())

            loss = criterion(output, labels)
            train_loss.append(loss.item())

            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            count = count + 1

            if (count % eval_every) == 0:
                #validate
                model.eval()
                early = 0
                track_acc = 0
                max_e = 1
                
                val_accs, val_losses, val_preds = run_eval(args, model, valid_loader)
                val_acc = np.mean(val_accs)

                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Acc: {:.6f}...".format(np.mean(train_accs)),
                    "Loss: {:.6f}...".format(np.mean(train_loss)))

                print("Val_Acc: {:.6f}...".format(val_acc),
                    "Val_Loss: {:.6f}...".format(np.mean(val_losses)))

                #early stopping
                if val_acc > track_acc:
                    track_acc = val_acc
                    max_e = e + 1
                    early = 0
                    #remove previous model
                    model_files = [f for f in listdir(args.trained_model_dir) if isfile(join(args.trained_model_dir, f))]
                    if model_files:
                        MODEL_PATH = join(args.trained_model_dir,model_files[0])
                        os.remove(MODEL_PATH)
                    #save model
                    torch.save(model.state_dict(), join(args.trained_model_dir,'checkpoint_'+str(e) +'.pt'))

                elif val_acc <= track_acc:
                    early = early + 1
                    if early >= 50: #max batches before stop
                        print("Max Accuracy ... {} at epoch {}".format(track_acc, max_e),
                            "Current Accuracy ...{} at epoch {} after {} epochs".format(val_acc, e, early))
                        break
            
                model.train()
        
        #check at the end of epoch
        if val_acc > track_acc:
            track_acc = val_acc
            max_e = e + 1
            #remove previous model
            model_files = [f for f in listdir(args.trained_model_dir) if isfile(join(args.trained_model_dir, f))]
            if model_files:
                MODEL_PATH = join(args.trained_model_dir,model_files[0])
                os.remove(MODEL_PATH)
            #save model
            torch.save(model.state_dict(), join(args.trained_model_dir,'checkpoint_'+str(e) +'.pt'))

def run_eval(args, model, valid_loader):
    val_accs=[]
    val_losses=[]
    val_preds=[]

    criterion = nn.NLLLoss()
    val_h = model.init_hidden(args.batch_size)
    model.eval()
    for inputs, labels in valid_loader:

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state
        val_h = tuple([each.data for each in val_h])

        # get the output from the model
        if(train_on_gpu):
            inputs = inputs.type(torch.cuda.LongTensor)
        else:
            inputs = inputs.type(torch.LongTensor)

        output, val_h = model(inputs, val_h)
        
        val_pred = torch.argmax(output, dim=1)
        val_acc = torch.sum(val_pred == labels)/args.batch_size
        val_accs.append(val_acc.item())

        val_loss = criterion(output, labels)
        val_losses.append(val_loss.item())
        val_accs.append(val_acc.item())
        val_preds.append(val_pred)

    return val_accs, val_losses, val_preds


def validate(args,model):

    #read data
    data = csv_reader(args.test_file,"test")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    reviews = [sub['review'] for sub in data]
    dev_x = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")['input_ids']
    dev_y = np.asarray([sub['rating'] for sub in data ])

    # dataloader
    logger.info("Data loader running")
    valid_data = TensorDataset(torch.LongTensor(dev_x), torch.from_numpy(dev_y))
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
    logger.info("Done")

    # move model to GPU, if available
    if(train_on_gpu):
        model.cuda()

    model.eval()
    val_accs, val_losses, val_preds = run_eval(args, model, valid_loader)
    
    print("Val_Acc: {:.6f}...".format(np.mean(val_accs)),
        "Val_Loss: {:.6f}...".format(np.mean(val_losses)))

    val_preds = [v.tolist() for v in val_preds]
    val_preds = [item for sublist in val_preds for item in sublist]
    evaluate(dev_y.tolist(), val_preds, "test")  
    out_path=join(args.output_path,'sota_output.csv')
    write_to_csv(out_path, reviews, val_preds,dev_y)

def infer(args, model):
    model.eval()

    def infer_helper(args,model):
        logger.info("Creating pseudo-labels")
        data = csv_reader(args.test_file,"inference")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
        reviews = [sub['review'] for sub in data]
        dev_x = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")['input_ids']
        #dev_x = torch.LongTensor(dev_x)

        # dataloader
        logger.info("Data loader running")
        valid_data = TensorDataset(torch.LongTensor(dev_x), torch.LongTensor(dev_x))
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
        logger.info("Done")

        # move model to GPU, if available
        if(train_on_gpu):
            model.cuda()

        val_preds=[]
        val_h = model.init_hidden(args.batch_size)
        model.eval()
        for inputs,_ in valid_loader:

            if(train_on_gpu):
                inputs, _ = inputs.cuda(), _

            # Creating new variables for the hidden state
            val_h = tuple([each.data for each in val_h])

            # get the output from the model
            if(train_on_gpu):
                inputs = inputs.type(torch.cuda.LongTensor)
            else:
                inputs = inputs.type(torch.LongTensor)

            output, val_h = model(inputs, val_h)
            val_pred = torch.argmax(output, dim=1)
            val_preds.append(val_pred)
            
        val_preds = [v.tolist() for v in val_preds]
        val_preds = [item for sublist in val_preds for item in sublist]
        out_path=join(args.output_path,'sota_output.csv')
        write_to_csv(out_path, reviews, val_preds)

    infer_helper(args, model)

    #train to get better psuedo-labels
    out_path=join(args.output_path,'sota_output.csv')
    shutil.move(out_path, join(args.data_path,'sota_output.csv'))
    data = raw_data_loader(args.data_path)
    train = data['train']
    val = data['dev']
    test = data['test']
    train.extend(data['inference'])
    #run in train mode
    args.mode='train'
    #create new temp model dirs
    #first remember the old one
    real_model_dir = args.trained_model_dir
    temp_model_dir = join(args.trained_model_dir,'temp')
    if not os.path.exists(temp_model_dir):
        os.makedirs(temp_model_dir)
    args.trained_model_dir = temp_model_dir
    preloaded = (train, val, test)
    trainer(args,model,preloaded)
  
    #delete first pseudo-labels from data_path
    os.remove(join(args.data_path,'sota_output.csv'))

    #final labels
    args.mode='inference'
    infer_helper(args,model)

    logger.info("Evaluating on pseudo-labels")
    #test with orginally trained model
    args.trained_model_dir = real_model_dir
    args.mode='test'
    args.test_file = out_path
    validate(args, model)

    #clean-up
    shutil.rmtree(temp_model_dir)
    

#Instantiate the model w/ hyperparams
def init_trainer(args):
    output_size = args.output_size
    hidden_dim = 2
    n_layers = 1
    net = FinetuneTransformer(output_size, hidden_dim, n_layers)
    return net
  
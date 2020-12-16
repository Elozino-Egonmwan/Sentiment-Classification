import csv
import os
import argparse
import tqdm
import sys
import re
import nltk
import pickle
import math
import numpy as np
import operator
import shutil
from string import punctuation
from collections import Counter
from functools import reduce
from os import listdir
from os.path import isfile, join, exists
from utils import csv_reader, pad_batch, logger, write_to_csv, evaluate
from nltk.metrics import accuracy

import tensorflow as tf
from tensorflow.contrib import layers, rnn

def neural_naiveBayes(args):  
    
    grouped_revs, freq_dist = naiveBayes(args)

    if args.mode == 'train':
        train_validate(args,grouped_revs, freq_dist)

    elif args.mode == 'test':
        test(args, grouped_revs, freq_dist)
    else:
        infer(args, grouped_revs, freq_dist)

def naiveBayes(args):
    #group reviews by ratings
    grouped_revs= group_reviews(args.data_path)

    #calc freq dist of vocab from all reviews
    all_reviews = reduce((lambda x,y: x+y), grouped_revs)[0]
    freq_dist = getVocabulary(processSent(all_reviews))  

    return grouped_revs, freq_dist

def train_validate(args,grouped_revs, freq_dist):
    train = pickle.load( open( join(args.data_path,'temp','train.pkl'), "rb" ) )
    valid = pickle.load( open( join(args.data_path,'temp','valid.pkl'), "rb" ) )
    
    data = {'train':train,'test':valid}

    all_features = {}
    all_labels = {}
    all_reviews={}

    for d in data:
        info = data[d]
        reviews = [sub['review'] for sub in info ]
        labels = [sub['rating'] for sub in info ]

        #estimate features using laplace naive bayes
        features = estimate_features(reviews, grouped_revs, freq_dist, args.max_words_len)

        all_features[d] =pad_batch(features.tolist(),args.batch_size)
        all_labels[d] = pad_batch(labels, args.batch_size)
        all_reviews[d] = pad_batch(reviews, args.batch_size)
        
    #train the features using gru-based linear regression model
    model_wrapper(args,all_features,all_labels,all_reviews, len(data['test']))

def test(args, grouped_revs, freq_dist):
    fileName = args.test_file
    test_data = csv_reader(fileName,"test")
    reviews = [sub['review'] for sub in test_data]
    features = {}
    labels ={}
    labels['test'] = [sub['rating'] for sub in test_data]
    features['test'] = estimate_features(reviews, grouped_revs, freq_dist, args.max_words_len)
    model_wrapper(args,features,labels,reviews)

def infer(args, grouped_revs, freq_dist):
    logger.info("Creating pseudo-labels")
    create_psuedo_labels(args, grouped_revs, freq_dist) #first inference

    #train to get better psuedo-labels
    #move to data_path -- this concatenates with train data 
    out_file = join(args.output_path,'sota_output.csv')
    shutil.move(out_file, join(args.data_path,'sota_output.csv'))

    #run in train mode
    args.mode='train'
    #create new temp model dirs
    #first remember the old one
    real_model_dir = args.trained_model_dir
    temp_model_dir = join(args.trained_model_dir,'temp')
    if not os.path.exists(temp_model_dir):
        os.makedirs(temp_model_dir)
        os.makedirs(join(temp_model_dir,'eval'))
    args.trained_model_dir = temp_model_dir
    neural_naiveBayes(args)

    #delete first pseudo-labels from data_path
    os.remove(join(args.data_path,'baseline_output.csv'))

    #final labels
    grouped_revs, freq_dist = naiveBayes(args)
    args.mode='inference'
    create_psuedo_labels(args, grouped_revs, freq_dist) #second inference

    logger.info("Evaluating on pseudo-labels")
    #test with orginally trained model
    args.trained_model_dir = real_model_dir
    args.mode='test'
    args.test_file = out_file
    neural_naiveBayes(args)

    #clean-up
    shutil.rmtree(temp_model_dir)


def create_psuedo_labels(args, grouped_revs, freq_dist):
    #prediction mode
    fileName = args.test_file
    test_data = csv_reader(fileName,"inference")
    reviews = [sub['review'] for sub in test_data]
    features = {}
    features['inference'] = estimate_features(reviews, grouped_revs, freq_dist, args.max_words_len)
    model_wrapper(args,features,reviews=reviews)

def estimate_features(reviews,grouped_revs, freq_dist,max_words_len):
    #predict ratings
    logger.info("Estimating features")
    features =[]
    
    for rev in tqdm.tqdm(range(len(reviews))):
    #for rev in range(1000):
        probs = []
        probs_len =[]
        for i in range(len(grouped_revs)):
            score = computeScore(reviews[rev],grouped_revs[i][0],len(freq_dist), max_words_len)
            probs_len.append(len(score))
            probs.append(score)
        features.append(probs)

    features= np.array(features)
    return features

'''prepare for Training, eval or testing'''  
def model_wrapper(args,features,labels=None, reviews=None, len_before_padding=None):     
    #get input functions
    if args.mode=="inference":      
        num_batches = int(len(features['inference'])/args.batch_size)  
        inp_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(features["inference"]).astype(dtype="float32")},      
        batch_size = args.batch_size,
        num_epochs=1,
        shuffle=False)

    else:
        num_batches = int(len(features['test'])/args.batch_size)
        inp_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(features["test"]).astype(dtype="float32")},      
        y=np.array(labels["test"]), 
        batch_size = args.batch_size,
        num_epochs=1,
        shuffle=False)

        if args.mode=="train":
            num_batches = int(len(features['train'])/args.batch_size)
            train_inp_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(features["train"]).astype(dtype="float32")},
            y=np.array(labels["train"]), 
            batch_size = args.batch_size,
            num_epochs=None,
            shuffle=True)

    #initialize estimator
    params ={"batch_size": args.batch_size, "hidden_dim":args.max_words_len, "num_classes":1, 'mode':args.mode}
    run_config = tf.estimator.RunConfig(tf_random_seed=args.random_seed, save_summary_steps=num_batches,save_checkpoints_steps=num_batches,keep_checkpoint_max=30,log_step_count_steps=num_batches)
    estimator = tf.estimator.Estimator(
        model_fn=linearRegression,
        model_dir=args.trained_model_dir,
        config = run_config,
        params = params) 

    # Set up logging for training
    # Log the values in the "predictions" tensor with label "pred"    
    tensors_to_log = {"pred": "predictions"}   
    print_predictions = tf.train.LoggingTensorHook(
            tensors_to_log,every_n_iter=num_batches*2)

    #sess = tf.InteractiveSession()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    #K.set_session(sess)

    '''run model'''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    tf.reset_default_graph() #reset graph before importing saved checkpoints
    
    early_stopping=tf.contrib.estimator.stop_if_no_increase_hook(estimator,metric_name='Accuracy',max_steps_without_increase=25*num_batches,run_every_steps=num_batches,run_every_secs=None)
    
    if args.mode=="train":
        epochs = args.epochs
        steps = num_batches * epochs 
        train_spec = tf.estimator.TrainSpec(input_fn=train_inp_fn,max_steps=steps,hooks=[early_stopping]) #hooks=[print_predictions]
        eval_steps= int(len(features["test"]))/args.batch_size
        eval_spec = tf.estimator.EvalSpec(input_fn=inp_fn,steps=eval_steps,throttle_secs=0, hooks=[print_predictions]) #hooks=[eval_print_predictions]
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        estimator.evaluate(input_fn=inp_fn)   
        infers = list(estimator.predict(input_fn=inp_fn))    
        print(infers)
        preds = [i["predictions"] for i in infers]
        out_path = join(args.output_path,'sota_dev.csv')
        evaluate(labels['test'][:len_before_padding],preds[:len_before_padding],'dev')
        write_to_csv(out_path,reviews['test'][:len_before_padding],preds[:len_before_padding],labels['test'][:len_before_padding])
        
    else:
        infers = list(estimator.predict(input_fn=inp_fn))    
        preds = [i["predictions"] for i in infers]
        out_path = join(args.output_path,'sota_output.csv')

        if args.mode=='test':
            estimator.evaluate(input_fn=inp_fn)
            evaluate(labels['test'],preds,'test')
            write_to_csv(out_path,reviews,preds,labels['test'])
        #inference
        else:
            write_to_csv(out_path,reviews,preds)
    
    coord.request_stop()
    coord.join(threads)  

def linearRegression(mode,features,labels,params):    
    inp = features["x"]  

    #encoder
    enc_cell = rnn.GRUCell(num_units=params['hidden_dim'])
    enc_out, enc_state = tf.nn.dynamic_rnn(enc_cell,inp,time_major=False,dtype=tf.float32)    
    inp = enc_out 
   
    weight, bias = weight_and_bias(params['hidden_dim'], params["num_classes"]) 
    logits=tf.squeeze(tf.map_fn(lambda x:tf.matmul(x, weight)+bias, inp))
    probs = tf.map_fn(lambda l: tf.nn.softmax(l), logits)
    preds =tf.map_fn(lambda x: tf.argmax(x), probs,dtype=tf.int64)
    tf.identity(preds, name="predictions") 
    infers = {"predictions":preds}

    if mode != tf.estimator.ModeKeys.PREDICT:
        loss= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
        tf.identity(loss, name="loss")

    else:
        spec = tf.estimator.EstimatorSpec(mode=mode,predictions=infers)
        return spec
        
    train_op = layers.optimize_loss(
    loss, tf.train.get_global_step(),
    optimizer='Adam',            
    learning_rate = 0.0001,
    clip_gradients=5.0)        
    
    tf.global_variables_initializer()
    tf.local_variables_initializer()
    tf.tables_initializer()

    spec = tf.estimator.EstimatorSpec(mode=mode,predictions=preds,
                                      loss = loss,
                                      eval_metric_ops={"Accuracy":tf.metrics.accuracy(labels,preds)},
                                      train_op=train_op)
        
    return spec    

def weight_and_bias(in_size, out_size):
    weight = tf.truncated_normal([in_size, out_size], stddev=0.01, dtype=tf.float32) 
    #weight = tf.zeros([in_size, out_size], dtype=tf.float32) 
    bias = tf.constant(0.1, shape=[out_size], dtype=tf.float32) #dtype=tf.float64
    return tf.Variable(weight,name="weights"), tf.Variable(bias,name="bias")

''' p(reviews|revPerLabel)
    reviews -- the reviews we want to rate
    rwr -- the reviews with (a specific) rating
'''
def computeScore(review,rwr, nVocab, max_words_len):
    rev_tokens = processSent(review)
    rwr_tokens = processSent(rwr)
    rwr_dist = getVocabulary(rwr_tokens) #frequency dist of tokens in rwr(reviews with a certain rating)
    rev_dist = freqRev(rev_tokens,rwr_dist) #freq dist of tokens in reviews we want to classify

    conditional_prob ={w: (float (rev_dist[w])/ (len(rwr_tokens) + nVocab)) for w in rev_dist}
    rev_counts = getVocabulary(rev_tokens)
    probs = [conditional_prob[w] ** rev_counts[w] for w in conditional_prob] #p(rev|rating)
    probs = pad_truncate(max_words_len, probs)
    
    return probs

def pad_truncate(max_words_len,vals):
    sent_len = len(vals)
    if sent_len < max_words_len:
        zeroes = list(np.zeros(max_words_len - sent_len, dtype=float))
        new = vals+zeroes
    else:
        new = vals[0:max_words_len] 
    return new

def getPriorProbs(grouped_revs):
    logger.info("Estimating Prior Probabilities")
    n_total_reviews = sum([len(rev) for rev in grouped_revs])
    prior_probs = [float(len(rev))/n_total_reviews for rev in grouped_revs]
    return prior_probs    

#for each word in the review, how many times do they occur in reviews with a specific rating
def freqRev(reviews,rwr_dist):
    reviews_vocab = set(reviews)
    reviews_dist={w:1+rwr_dist[w] if w in rwr_dist.keys() else 1 for w in reviews_vocab} #smoothing the frequencies
    reviews_dist = dict( sorted(reviews_dist.items(), key=operator.itemgetter(1),reverse=True)) #sort
    return reviews_dist

#count of each word in the list of tokens eg {'the':2}   
def getVocabulary(tokens):  
    fdist = Counter(tokens)
    return fdist

def processSent(sentence):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    stop_words= ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "''",'""',"'"]
    #ensure puncts are separated from words
    sent = re.findall(r"[\w']+|[.,!?;]", sentence)
    #remove puncts/numbers/stopwords and change to lowercase
    sent = [c.lower() for c in sent if (c not in punctuation) and (c not in stop_words) and (not is_number(c))]
    #sent = [c.lower() for c in sent if not is_number(c)]
    return sent

#group reviews -- by ratings
def group_reviews(data_path):
    logger.info('Grouping reviews')
    data_files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and '_test.' not in f]

    #augument training data
    dev_file =[f for f in data_files if '_dev' in f]
    if dev_file:
        dev = join(data_path,dev_file[0])
        new_file_name = dev_file[0].replace('dev','temp')
        new_dev = join(data_path,new_file_name)
        shutil.copyfile(dev,new_dev)
        data_files.append(new_file_name)

    labels_1 = []
    labels_2 = []
    labels_3 = []
    labels_4 = []
    labels_5 = []

    train =[]
    validation = []

    total_valid_reviews = 0 #across all sets

    for file_ in data_files:
        file_path = join(data_path,file_)
        logger.info('Loading dataset from %s', file_path)
        marker = 0

        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if marker == 0: #first line contains headings or keys
                    keys = row
                    marker = marker + 1
                else:
                    try:
                        #skip inValid rows -- rinvalid rating or empty review
                        if int(row[2]) < 0 or int(row[2]) > 5 or row[1]=="":
                            continue

                        else:
                            rating = int(row[2]) - 1

                            #group by labels
                            if rating == 1:
                                labels_1.append(row[1])
                            elif rating == 2:
                                labels_2.append(row[1])
                            elif rating == 3:
                                labels_3.append(row[1])
                            elif rating == 4:
                                labels_4.append(row[1])
                            else:
                                labels_5.append(row[1])

                            #group by datasets
                            data = {'id': row[0], 'review':row[1], 'rating': rating}
                            if '_train' in file_path:
                                train.append(data)
                            elif '_dev' in file_path:
                                validation.append(data)
                            else:
                                train.append(data)
                            line_count += 1

                    except Exception as e: 
                        print(e)
                        continue

            logger.info(f'{line_count} lines in {file_path}.')
            total_valid_reviews += line_count

    #sanity check
    n_all_reviews = len(labels_1) + len(labels_2) + len(labels_3) + len(labels_4) + len(labels_5)
    if total_valid_reviews != n_all_reviews:
        print(f"Error processing: {n_all_reviews} not equal to {total_valid_reviews}")
        sys.exit()
    else:
        nl = '\n'
        print(f"Label 1: {len(labels_1)}{nl}"
        f"Label 2: {len(labels_2)}{nl}"
        f"Label 3: {len(labels_3)}{nl}"
        f"Label 4: {len(labels_4)}{nl}"
        f"Label 5: {len(labels_5)}{nl}")

    #save reviews
    temp_dir = join(data_path,'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(join(data_path,'temp'))
    pickle.dump(train, open(join(data_path,'temp','train.pkl'), "wb" ))
    pickle.dump(validation, open(join(data_path,'temp','valid.pkl'), "wb" ))
    
    #clean_up
    if dev_file:
        os.remove(new_dev)
        
    return [labels_1, labels_2, labels_3, labels_4, labels_5]

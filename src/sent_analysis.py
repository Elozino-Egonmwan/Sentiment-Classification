import argparse
from os.path import exists
from utils import logger,init_logger
from sota import finetune_mrpc
from baseline import neural_naiveBayes

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Sentiment Analysis')
    parser.add_argument('-model', default='sota', type=str, choices=['baseline', 'sota'])
    parser.add_argument('-mode', default='test', type=str, choices=['train', 'test', 'inference'], help='test on any labelled data; infer on uunlabelled data')
    parser.add_argument('-data_path', default='data',type=str, help='path to the folder cotaining datasets')
    parser.add_argument('-test_file', default='data/sentiment_dataset_dev.csv', type=str, help='path to file')
    parser.add_argument('-output_path', default='output', type=str, help='path to store the output')
    parser.add_argument('-log_file', default='logs',type=str, help='path to save logs') 

    parser.add_argument('-max_sent_len', default=6, type=int, help='maximum number of sentences in each review')        
    parser.add_argument('-max_word_sent_len', default=19, type=int, help='maximum number of words per sentence in each review') 
    parser.add_argument('-max_words_len', default=140, type=int, help='maximum number of words in each review')

    parser.add_argument('-trained_model_dir', default='models/sota', type=str, help='directory to trained model') 
    parser.add_argument('-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-gradient_clipping', default=5, type=int, help='clip gradient at')                            
    parser.add_argument('-batch_size', default=8, type=int, help='training batch size') 
    parser.add_argument('-random_seed', default=666, type=int, help='seed value')        
           
    parser.add_argument('-hidden_dim', default=300, type=int, help='hidden dimension')   
    parser.add_argument('-n_layers', default=2,type=int, help='number of nn layers')                                                                                                                       
    parser.add_argument('-output_size', default=5, type=int, help='number of classes')                                                             
    parser.add_argument('-learning_rate', default=0.001, type=float, help='learning rate')

    args = parser.parse_args()
    assert exists(args.data_path)
    
    if args.model == 'baseline':
        args.log_file = 'logs/baseline_logs.txt'
        args.trained_model_dir = 'models/baseline'
        args.batch_size=50
        init_logger(args.log_file)
        neural_naiveBayes(args)

    else:
        args.log_file = 'logs/sota_logs.txt'
        args.trained_model_dir = 'models/sota'
        init_logger(args.log_file)
        finetune_mrpc(args)

        
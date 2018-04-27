import json
import logging
from log import init_console_logger
import argparse
import os
from  nltk import ngrams
import multiprocessing as mp


LOGGER = logging.getLogger('bias scores')
LOGGER.setLevel(logging.DEBUG)


DEFAULT_MALE_NOUNS = {
    'gentleman', 'man', 'men', 'gentlemen', 'male', 'males', 'boy', 'boyfriend','mr',
    'boyfriends', 'boys', 'he', 'his', 'him', 'husband', 'husbands', 'son' , 'sons'
}

DEFAULT_FEMALE_NOUNS = {
    'woman', 'women', 'ladies', 'female', 'females', 'girl', 'girlfriend', 'ms','mrs',
    'girlfriends', 'girls', 'her', 'hers', 'lady', 'she', 'wife', 'wives', 'daughter', 'daughters'
}

def sortbybias(d):
    
    d_s = sorted(d.items(), key = lambda t: t[1] )
    return d_s

def gender_ratios_m_f(output_data_dir,file):
    n = 0
    tot = 0 
    print("Gender Ratios...")
    with open(file,'r') as f:
        data = json.load(f)
    bias_record = {}
    for words in data:
        if (data[words]['m']+data[words]['f']!=0 and data[words]['f']!=0 and data[words]['m']!=0):
            score = data[words]['m']/(data[words]['m']+data[words]['f'])
            tot+=score
            n +=1
            rec = {"b_score" : score}
            data[words].update(rec)
            bias_record[words] = json.dumps(data[words])
    print(bias_record)
    print(sortbybias(bias_record))
    output_file = os.path.join(output_data_dir, 'biased_words_m_f')   
    print("Bias_score: ", (tot/n))
    with open(output_file,'w') as fp:
        json.dump(bias_record,fp, sort_keys=True)   



def gender_ratios(output_data_dir,file):
    print("Gender Ratios...")
    with open(file,'r') as f:
        data = json.load(f)
    bias_record = {}
    for words in data:
        if (data[words]['m']+data[words]['f']!=0):
            score = data[words]['m']/(data[words]['m']+data[words]['f'])
            rec = {"b_score" : score}
            data[words].update(rec)
            bias_record[words] = json.dumps(data[words])
    print(bias_record)
    print(sortbybias(bias_record))
    output_file = os.path.join(output_data_dir, 'biased_words')    
    with open(output_file,'w') as fp:
        json.dump(bias_record,fp, sort_keys=True)   


def get_cooccurrences(file, data, window):           
    
   
    with open(file, 'r') as fp:
        print(fp)
        sentences = fp.read()

    male_nouns = DEFAULT_MALE_NOUNS
    female_nouns = DEFAULT_FEMALE_NOUNS
    n_grams = ngrams(sentences.split(), window)
    
    for grams in n_grams:
        pos = 1
        m = 0 
        f = 0 
        for w in grams:
                pos+=1
                if w not in data:
                    data[w]= {"m":0, "f":0}
                
                if pos==int((window+1)/2):
                    if w in male_nouns:
                        m = 1
                    if w in female_nouns:
                        f = 1
                    if m > 0:
                        for t in grams:
                            if t not in data:
                                data[t]= {"m":0, "f":0}
                            data[t]['m']+=1
                    if f > 0:
                        for t in grams:
                            if t not in data:
                                data[t]= {"m":0, "f":0}
                            data[t]['f']+=1
    return data
    

def coccurrence_counts(dataset_dir, output_dir, window=7,num_workers=1):
    
    
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = os.path.abspath(output_dir)
    output_data_dir = os.path.join(output_dir, 'bias_scores')
    
    if not os.path.isdir(dataset_dir):
        raise ValueError('Dataset directory {} does not exist'.format(dataset_dir))

    if not os.path.isdir(output_data_dir):
        os.makedirs(output_data_dir)
        
    data ={}
    worker_args = []
    LOGGER.info("Getting list of files...")
    for root, dirs, files in os.walk(dataset_dir):
        root = os.path.abspath(root)
        for fname in files:
            basename, ext = os.path.splitext(fname)
            if basename.lower() == 'readme':
                continue
            txt_path = os.path.join(root, fname)
            data = get_cooccurrences(txt_path, data, window )
    output_file = os.path.join(output_data_dir, 'all_words')     

    with open(output_file,'w') as fp:
        json.dump(data,fp)
    
    gender_ratios(output_data_dir,output_file)  
    gender_ratios_m_f(output_data_dir,output_file) 

            
            
def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Get the bias scores of a given text file')
    parser.add_argument('dataset_dir', help='Path to directory containing text files', type=str)
    parser.add_argument('output_dir', help='Path to output directory', type=str)
    parser.add_argument('-n', '--num-workers', dest='num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('-w', '--window', dest='window', type=int, default=10, help='Context Window')
    return vars(parser.parse_args())


if __name__ == '__main__':
    init_console_logger(LOGGER)
    coccurrence_counts(**(parse_arguments()))
            
    
    

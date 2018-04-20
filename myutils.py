#Build word2vec model from glove txt vector file


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sys import stdout
import numpy as np
from matplotlib import pyplot
import sys
from preprocess import *
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.models import word2vec
import glob
 
import os

def build_w2v(glove_input_file):
    word2vec_output_file = 'tmp.txt.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    return model

#Build visualisations
#copied from Kaggle notebook
def tsne_plot1(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    
    
    
def calculate_bias(male, female, word_list, model):
    male_v = model.get_vector(male)
    female_v = model.get_vector(female)
    diff2 = male_v-female_v

    print(male_v, 1 - spatial.distance.cosine(man_v, diff2))
    print(female_v, 1 - spatial.distance.cosine(woman_v, diff2))

    for w in word_list:
         print (w, (1 - spatial.distance.cosine(model.get_vector(w),diff2)))
            
            
def bias_axis(male, female):
    return (male_v - female_v)


def equalize(male, female,model, male_bias, female_bias):
    male_v = model.get_vector(male)
    female_v = model.get_vector(female)
    
    male_bias_v = model.get_vector(male_bias)
    female_bias_v = model.get_vector(female_bias)
    
    diff2 = male_v-female_v
    
    v = (male_v+female_v)/2
    
    diff = bias_axis(male_bias_v, female_bias_v)
    
    vx= (diff*(v.dot(diff))/diff.dot(diff))
    vy = v -vx
    
    mx = (diff*(man.dot(diff))/diff.dot(diff))
    wx = (diff*(woman.dot(diff))/diff.dot(diff))
    
    mc = (np.sqrt(abs(1-vy.dot(vy)))*(mx-vx))/np.sqrt((man-vy-vx).dot((man-vy-vx)))
    wc = np.sqrt(abs(1-vy.dot(vy)))*(wx-vx)/np.sqrt((woman-vy-vx).dot((woman-vy-vx)))

    eq_man = mc+vy
    eq_woman=wc+vy
    
    print("Cosine Distance between male and bias vectors: ",1 - spatial.distance.cosine(male, diff))
    print("Cosine Distance between female and bias vectors: ",1 - spatial.distance.cosine(female, diff))

    print("Cosine Distance between equalized male and bias vectors:",1 - spatial.distance.cosine(eq_male, diff))
    print("Cosine Distance between equalized female and bias vectors: ",1 - spatial.distance.cosine(eq_female, diff))
    




def build_glove_dictionary(glove_file):
    
    print ('building glove dictionary...')
    #glove_file = 'vectors_brown_100.txt'
    glove_dict = {}
    with open(glove_file) as fd_glove:
        j=0
        for i, input in enumerate(fd_glove):
            input_split = input.split(" ")
            #print input_split
            key = input_split[0] #get key
            del input_split[0]  # remove key
            j+=1
            stdout.write("\rloading glove dictionary: %d" % j)
            stdout.flush()
            values = []
            for value in input_split:
                values.append(float(value))
            np_values = np.asarray(values)
            glove_dict[key] = np_values
            #else:
                #print key
    print ("")
    print ('dictionary build with length', len(glove_dict))

    return glove_dict

def build_glove_matrix(glove_dictionary):
    """
        return word2idx and matrix
    """
    idx2word = {}
    glove_matrix = []
    i=0
    for key, value in glove_dictionary.items():
        idx2word[i] = key
        glove_matrix.append(value)
        i+=1
    return np.asarray(glove_matrix), idx2word

def check_similarity(glove_matrix, word):
    return cosine_similarity(word.reshape(1, -1), glove_matrix)

def build_matrix_to_tsne(glove_dict, tokens):
    matrix = []
    for token in tokens:
        if token in glove_dict:
            matrix.append(glove_dict[token])
    return matrix


def tsne_plotly(glove_file, words):

    #words = []
#if len(sys.argv)<2:
    #print ('Words not specified')
    
#else:
#    for i in range(1, len(sys.argv)):
#        words.append(sys.argv[i])
    glove_file = os.path.join('glove',glove_file)
    print(glove_file)
    print ('Words that will be used', words)
 
    glove_dict = build_glove_dictionary(glove_file)
    glove_matrix, idx2word = build_glove_matrix(glove_dict)
    model = TSNE(n_components=2, random_state=0)

    to_plot = []
    labels = []
    not_found = 0
    len_words = len(words)
    for word in words:
        try:
            cosine_matrix = check_similarity(glove_matrix, glove_dict[word])
            ind = cosine_matrix[0].argsort()[-100:][::-1]
            closest = ind.tolist()
            tokens = [idx2word[idx] for idx in closest]
            to_reduce = build_matrix_to_tsne(glove_dict, tokens)
            #print to_reduce.shape
            labels += [token for token in tokens]
            to_plot += [x_y for x_y in to_reduce]
        except:
            len_words-=1
            print ('Word not found', word)

    print (len_words)
#print to_plot.shape
#print to_plot
    X_hdim = np.array(to_plot)
#print X_hdim
    print (X_hdim.shape)
    X = model.fit_transform(X_hdim)
    X_x = np.zeros((len_words*10, 2))
    labels_x = []
    print (X.shape)
    k=0
    ranges = [x*100 for x in range (0, len_words)]
    print (ranges)
    for i in ranges:
        for j in range(1, 11):
            print (i+j-1, k)
            X_x[k] = X[i+j-1]
            k+=1
            labels[i+j-1]
            labels_x.append(labels[i+j-1])


    print (labels_x)
    print (X_x.shape)
    pyplot.figure(figsize = (70,70))
    pyplot.scatter(X_x[:,0],X_x[:,1])
    for i, label in enumerate(labels_x):
        pyplot.annotate(label, (X_x[i,0],X_x[i,1]))
    pyplot.show()
    
    
    

def write_to_json(sentences, target_pos = None):
    data ={}
    
    if not target_pos:
        target_pos = DEFAULT_TARGET_POS
        
    words ={}    
    
    index = 0
    for sentence in sentences:
        w_string = ' '
        
        for w in sentence:
            w_string += ' '+ w
        print(w_string)
        
        data[index]={}
        va =[]
        adja = []
        adva =[] 
        for w in en(w_string):
            #print(w.text, w.pos_)
            if (w.pos_) == 'VERB':
                #print("shikha")
                #words[w.text]
                va.append(w.text)
                #print ("va:   ", va)
                tmp = { w.pos_ : va }
                #print(tmp)
                data[index].update(tmp) 
                #print(index, ":", data[index],)
            
            if (w.pos_) == 'ADJ':
                adja.append(w.text)
                #print(adja)
                tmp  = { w.pos_ : adja }
                data[index].update(tmp) 
                
                
            if (w.pos_) == 'ADV':
                adj = w.text
                adva.append(w.text)

                #print(adva)
                tmp = { w.pos_ : adva }
                data[index].update(tmp) 
                
        #print(data)
        
        
        
        index+=1    
        
        
    #print(data)  
    
    with open('data.json','w') as fp:
        json.dump(data,fp)                        
      









              
from sklearn.decomposition import PCA

def normalize(model):
    
    words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
    vecs = [model[w] for w in words]
    vecs = np.array(vecs)
    print(vecs.shape)
    words = words
    index = {w: i for i, w in enumerate(words)}
    norms = np.linalg.norm(vecs, axis=1)
    
    return 



def debias(model, defining_sets):
    
    d=[]
    n = len(defining_sets)
    for s in defining_sets:
        D = int(len(s))
        vec_mid = np.array(get_unit_vector(model.get_vector(s[0]))-get_unit_vector(model.get_vector(s[1])))/D
              
        d = np.append(d, get_unit_vector(model.get_vector(s[0]) - vec_mid))
        d = np.append(d, get_unit_vector(model.get_vector(s[1]) - vec_mid))
        #print(len(d))
    m  = int(len(d)/(2*n))
    C=d.reshape(m,2*n)
    #print(C.shape)

    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(C) 
    #print(pca.explained_variance_ratio_)
    #print(pca.singular_values_)
    bias_v = pca.components_

    
def neutralize(v, bias_v):
    v_neutral = v - v*v.dot(bias_v)/(bias_v.dot(bias_v))
    return v_neutral
    
    
   
    
def equalize(equality_set, bias_v):
    E  = int(len(equality_set))
    mid = []
    for s in equality_set:
        mid += []
    mid = mid/E
    
    for s in equality_set:
        v = s - mid*mid.dot(bias_v)/(bias_v.dot(bias_v))
        s = v  + (s-v)*math.sqrt(1-v.dot(v))/(s-v).dot(s-v)
        
    
  

       
       
       


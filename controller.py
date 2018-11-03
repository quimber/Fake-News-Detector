from model import *
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config

from operator import itemgetter
from keras.models import load_model
import pandas as pd
import sys

#Data Preperation


df = pd.read_csv('sample_data.csv')


sentences1 = list(df['sentences1'])
sentences2 = list(df['sentences2'])
is_similar = list(df['is_similar'])
del df



# Word Embedding 



# creating word embedding  
tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  siamese_config['EMBEDDING_DIM'])

embedding_meta_data = {
	'tokenizer': tokenizer,
	'embedding_matrix': embedding_matrix
}

# creating sentence pairs
sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
del sentences1
del sentences2



# Training 

from config import siamese_config


class Configuration(object):
    """Dump stuff here"""

CONFIG = Configuration()
CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, 
					    CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio)

best_model_path = siamese.train_model(sentences_pair, is_similar, embedding_meta_data, model_save_directory='./')


qp=0.52
# Testing 
model = load_model(best_model_path)

def output(l1,l2):
    test_sentence_pairs = test_sentence_pairs = [(l1,l2)]
    # 
    test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])
    # 
    preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
    results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
    results.sort(key=itemgetter(2), reverse=True)
    return results[0][2]

def paragraph_sim(para1,para2):
    list1=para1.split(".")
    list2=para2.split(".")
    
    deflt=60
    i=0
    j=0
    summ=0
    for i in range(len(list1)):
        maxm=0
        for j in range(len(list2)):
            
                
            a=output(list1[i][1:],list2[j][1:]) #calling/importing the similarity checker part. Function of the final_model file
        
            if(a>maxm):
                maxm=a
                
        summ+=maxm
    return summ/len(list1)




def function(s):
    from googlesearch import search
    j=0
    list=[]
    for url in search(s+"BBC", tld='com.pk', lang='es', stop=5):
        list.append(url)
        j=j+1
    return list[0]



paraalpha=raw_input("Enter a news article: ")
from newsplease import NewsPlease
article = NewsPlease.from_url(function(paraalpha))

parabeta=article.text
y=paragraph_sim(paraalpha,parabeta)
print y
'''def web_scraping(input_paragraph):
    from googlesearch import search
    s = input_paragraph
    j=0
    list=[]
    for url in search(s+"BBC", tld='com.pk', lang='es', stop=5):
        list.append(url)
        j=j+1
    list[0]
    from newspaper import Article
    url=list[0]
    a=Article(url, language='en')
    a.download
    a.parse
    para=[]
    para=a.text
    return para'

paragraph = "tiger is national animal of india"
actual_news=web_scraping(paragraph)
print(actual_news)
result_score=paragraph_sim(paragraph,actual_news)'''
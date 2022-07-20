import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import string

model = pickle.load(open('/Users/ishanranasinghe/Desktop/Fake_News_Test_Master/model_xgboost.pkl','rb'))
vectorizer = pickle.load(open('/Users/ishanranasinghe/Desktop/Fake_News_Test_Master/vectorizer.pkl','rb'))

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words


for i in range(10):
    df = pd.DataFrame(columns=['content'])
    df['content'] =[input("Please enter the content: ")]
    df['content'].apply(process_text)
    text = vectorizer.transform(df['content'])
    pred = model.predict(text)

    label = ''
    if pred[0] == 0:
       label = "Real"
    else:
        label = "Fake"

    print("The prdeictions for your data is: {}".format(label))
    print("=======================================================")


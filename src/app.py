### NLP and SVM

# Import libraries

import pandas as pd
import pickle
import numpy as np
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


## Load dataset and do the necessary transformations

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')

df_interin = df_raw.copy()

# Remove duplicates

df_interin = df_interin.drop_duplicates().reset_index(drop = True)

# functions to clean the text

def comas(text):
    """
    Elimina comas del texto
    """
    return re.sub(',', ' ', text)

def espacios(text):
    """
    Elimina enters dobles por un solo enter
    """
    return re.sub(r'(\n{2,})','\n', text)

def minuscula(text):
    """
    Cambia mayusculas a minusculas
    """
    return text.lower()

def numeros(text):
    """
    Sustituye los numeros
    """
    return re.sub('([\d]+)', ' ', text)

def caracteres_no_alfanumericos(text):
    """
    Sustituye caracteres raros, no digitos y letras
    Ej. hola 'pepito' como le va? -> hola pepito como le va
    """
    return re.sub("(\\W)+"," ",text)

def comillas(text):
    """
    Sustituye comillas por un espacio
    Ej. hola 'pepito' como le va? -> hola pepito como le va?
    """
    return re.sub("'"," ", text)

def palabras_repetidas(text):
    """
    Sustituye palabras repetidas

    Ej. hola hola, como les va? a a ustedes -> hola, como les va? a ustedes
    """
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

def esp_multiple(text):
    """
    Sustituye los espacios dobles entre palabras
    """
    return re.sub(' +', ' ',text)


def url(text):
    """
    Remove https
    """
    return re.sub(r'(https://www|https://)', '', text)


# clean url
df_interin['url_limpia'] = df_interin['url'].apply(url).apply(caracteres_no_alfanumericos).apply(esp_multiple)

# Transform target variable
df_interin['is_spam'] = df_interin['is_spam'].apply(lambda x: 1 if x == True else 0)


## NLP techniques to preprocess data before modeling

df = df_interin.copy()

X = df['url_limpia']

y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y, random_state=2207)

vec = CountVectorizer()

X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()


## SVM to construct a classifier for URLs

# No hyperparameter tuning

classifier = SVC(C = 1.0, kernel = 'linear', gamma = 'auto')
classifier.fit(X_train, y_train)


# Hyperparameter tuning

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(SVC(random_state=1234),param_grid,verbose=2)

grid.fit(X_train,y_train)


# Performance (accuracy and f1 for both classes) was better in the first classifier
# Save that one as best model

# Save it as best model

best_model = classifier

# Save it for future use

pickle.dump(best_model, open('../models/best_model.pickle', 'wb')) # save the model
# modelo = pickle.load(open('../models/best_model.pickle', 'rb')) # read the model in the future
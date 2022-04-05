"""
INFO: This file contains the method needed to make predictions
AUTHOR: Siqi-Fang
DATE : Last Update 03-29-2022
"""

import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import plotly
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
import pickle
import bz2
#from memory_profiler import profile  # dont put in requirements

sentiments = ["Negative", "Positive"]
STOPWORDS = []

tvec = bz2.BZ2File('ml_models/tvec.pbz2', 'rb')
tvec = pickle.load(tvec)
# Load Model
with open('ml_models/logReg.sav', 'rb') as f:
    model = pickle.load(f)
analyze = make_pipeline(tvec, model)

with open('eng_stopword.txt', 'r') as f:
    for word in f:
        word = word.split('\n')
        STOPWORDS.append(word)



def twit_preproc(text, tokenized=False):
    """
    Perform preprocessing, steps include
        clean special text --> tokenize
        --> remove stop words(nltk stopwords)
        --> lemmatize --> join text(optional)
        Parameters:
            text(str): input text
            tokenized(bool): If True, a list of words will be returned
    """

    def clean_text(text):
        """Make text lowercase, remove text in square brackets,
        remove links, punctuation and words containing numbers."""
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('<.*?>+', '', text)  # remove text in brackets
        text = re.sub('https?://\S+|www\.\S+', '', text)  # remove link
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)  # remove numbers
        text = re.sub('\w*\d\w*', '', text)
        return text

    # Tokenize & remove stop words
    tokenizer = RegexpTokenizer(r'\w+')
    text_arr = tokenizer.tokenize(clean_text(text))
    text_arr = [y for y in text_arr if y not in STOPWORDS]

    # lemmatize and join the words
    lemmatizer = WordNetLemmatizer()
    text_arr = [lemmatizer.lemmatize(x) for x in text_arr]

    # join the text
    if not tokenized:
        text = " ".join(x for x in text_arr)

    return text

def single_prediction(t):
    """ Returns the result of the sentiment analysis of input text
        Parameters:
            t(string): text input
        Returns:
            sentiment(int): 0 -- Negative
                            4 -- Positive
            confidence(float): the probability that the word belongs to the sentiment
    """
    clean = twit_preproc(t)
    X = tvec.transform([clean])
    sentiment = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])
    return sentiment, confidence


def _single_explanation(text, features):
    """
    Provides Interpretation of the analysis of the text:
    A list of tuples containing,
    (<the feature/word>, <importance of the feature>)
    Parameters:
            text(str): text to analyze
            features(int): maximum number of feature to include in the explanation
    Returns:
        explanation: dictionary contains 4 lists, word with neg and pos sentiment and their
        respective importance
    """
    explanation = {
        'p_words': [],
        'p_importance': [],
        'n_words': [],
        'n_importance': [],
    }

    explainer = lime_text.LimeTextExplainer(class_names=sentiments)
    exp = explainer.explain_instance(text, analyze.predict_proba, num_features=features).as_list()

    # generate a list of positive and negative features with abs(importance) > 0.05
    # maybe this threshold should be adjusted by words in text
    for word, score in exp:
        if score > 0.05:
            explanation['p_words'].append(word)
            explanation['p_importance'].append(round(float(score), 3))
        elif score < -0.05:
            explanation['n_words'].append(word)
            explanation['n_importance'].append(round(float(score), 3))

    return explanation


def plot_explanation(text, features=6):
    """
    Plot the explanations in a bar graph and highlight text with high importance by thier sentiment.
        Parameters:
            features(int): maximum number of feature to include in the explanation
            text(str): corpus to explain
        Returns:
            graph(str): <div> containing the graph
            text(str): highlighted text
    """
    exp = _single_explanation(text, features)
    # plot the bar graph
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=exp['n_importance'], y=exp['n_words'], name="Negative", orientation='h', marker=dict(color='#CD5C5C')))
    fig.add_trace(go.Bar(
        x=exp['p_importance'], y=exp['p_words'], name="Positive", orientation='h', marker=dict(color='#008000')))
    fig.update_layout(
        title="Feature Importance for the Tweet",
        xaxis_title="Word Importance",
        yaxis_title="Word",
        legend_title_text="Word Sentiment"
    )
    # save the graph as <div> html element
    graph = plotly.offline.plot(figure_or_data=fig, output_type='div', auto_open=False)

    # highlight the words with high importance by their sentiment
    arr = text.split()
    for i, word in enumerate(arr):
        word = re.compile(r"\w*").match(word).group()
        if word in exp['p_words']:
            arr[i] = '<span style="background-color:#008000;color:#FFF;">' + arr[i] + '</span>'
        elif word in exp['n_words']:
            arr[i] = '<span style="background-color:#CD5C5C;color:#FFF;">' + arr[i] + '</span>'
    text = " ".join(arr)

    return graph, text

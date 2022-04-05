from memory_profiler import profile
@profile
def im():
    import re
    import string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
    from sklearn.pipeline import make_pipeline
    from lime import lime_text
    from lime.lime_text import LimeTextExplainer
    import plotly
    import plotly.graph_objects as go
    import pickle
    import bz2

    tokenizer = RegexpTokenizer(r'\w+')


from memory_profiler import profile

STOPWORDS = []
with open('eng_stopword.txt', 'r') as f:
    for word in f:
        word = word.split()[0]
        STOPWORDS.append(word)
print(STOPWORDS)
#if __name__ == '__main__':

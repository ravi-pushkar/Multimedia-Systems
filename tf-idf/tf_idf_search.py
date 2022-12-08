from re import T
from tkinter.tix import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
from extract import getText
from clean import TextCleaner
import pickle

class TfIdfSearchEngine():
    def __init__(self) -> None:
        return
    
    def build(self, articleList) -> None:

        self.vectorizer = TfidfVectorizer()
        self.tc = TextCleaner()
        self.articles = [] # (Article name, full text)
        corpus = []
        for article in articleList:
            full_text = getText(article)
            if full_text is None:
                print(f'[Error] Could not find article for {article}...')
                continue
            self.articles.append((article, full_text))
            corpus.append(self.tc.cleanText(full_text))

        st = time.time()
        self.term_matrix = self.vectorizer.fit_transform(corpus)
        ed = time.time()
        print(f'Generating doc matrix took {ed - st} s...')


    def save(self, save_file_name='tf_idf_se.obj'):
        f = open(save_file_name, 'wb')
        pickle.dump((self.vectorizer, self.articles, self.tc), f)
        f.close()

    def load(self, save_file_name='tf_idf_se.obj'):
        f = open(save_file_name, 'rb')
        obj = pickle.load(f)
        self.vectorizer = obj[0]
        self.articles = obj[1]
        self.tc = obj[2]
        f.close()

    
        
    def getSearchResults(self, query, max_results=10):
        query = self.tc.cleanText(query)
        query_vec = self.vectorizer.transform([query])
        results = cosine_similarity(self.term_matrix, query_vec).reshape((-1,))
        for i in results.argsort()[::-1][:max_results]:
            print(self.articles[i][0])
            




tf_idf_se = TfIdfSearchEngine()


tf_idf_se.build(['TaylorSwift', 'DuaLipa', 'ElonMusk', 'Virat Kohli', 'Quantum Mechanics', 'Linear Algebra', 'India'])
tf_idf_se.save('tf_idf_se.obj')
tf_idf_se.load('tf_idf_se.obj')


tf_idf_se.getSearchResults('study', 10)


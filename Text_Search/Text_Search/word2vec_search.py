from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import time
import numpy as np
from extract import getText
from clean import TextCleaner
import pickle

class Word2VecSearchEngine():
    def __init__(self) -> None:
        self.embeddingSize = 1000
    
    def build(self, articleList) -> None:


        self.tc = TextCleaner()
        self.articles = [] # (Article name, full text)
        train_data = []
        for article in articleList:
            full_text = getText(article)
            if full_text is None:
                print(f'[Error] Could not find article for {article}...')
                continue
            self.articles.append((article, full_text))
            train_data.append(self.tc.cleanText(full_text).split())


        st = time.time()
        self.word2vec_model = Word2Vec(train_data, vector_size=self.embeddingSize, window=5, sg=1)
        ed = time.time()
        print(f'Training model took {ed - st} s...')

        self.docEmbeddings = []

        for (article, full_text) in self.articles:
            self.docEmbeddings.append(self.getEmbedding(full_text).tolist())
        
        self.docEmbeddings = sparse.csr_matrix(np.array(self.docEmbeddings))


        
    def getEmbedding(self, text):
        cleaned_text = self.tc.cleanText(text).split()
        word_embeddings = []
        if len(cleaned_text) == 0:
            return np.zeros(np.zeros(self.embeddingSize))
        for word in cleaned_text:
            if word in self.word2vec_model.wv.index_to_key:
                word_embeddings.append(self.word2vec_model.wv.word_vec(word))
            else:
                word_embeddings.append(np.random.rand(self.embeddingSize))
        
        return np.mean(word_embeddings, axis=0)
    
            
    def save(self, save_file_name='word2vec_se.obj'):
        f = open(save_file_name, 'wb')
        pickle.dump((self.word2vec_model, self.articles, self.tc, self.docEmbeddings), f)
        f.close()

    def load(self, save_file_name='word2vec_se.obj'):
        f = open(save_file_name, 'rb')
        obj = pickle.load(f)
        self.word2vec_model = obj[0]
        self.articles = obj[1]
        self.tc = obj[2]
        self.docEmbeddings = obj[3]
        f.close()

        
    def getSearchResults(self, query, max_results=10):
        # print("****** " , len(self.word2vec_model.wv.index_to_key))
        query_vec = sparse.csr_matrix(self.getEmbedding(query))
        results = cosine_similarity(self.docEmbeddings, query_vec).reshape((-1,))

        idx = 0
        for i in results.argsort()[::-1][:max_results]:
            idx += 1
            print(f'[{idx}] {self.articles[i][0]}')
            




# word2vec_se = Word2VecSearchEngine()


# word2vec_se.build(['TaylorSwift', 'DuaLipa', 'ElonMusk', 'Virat Kohli', 'Quantum Mechanics', 'Linear Algebra', 'India'])
# word2vec_se.save('word2vec_se.obj')
# word2vec_se.load('word2vec_se.obj')


# word2vec_se.getSearchResults('study', 10)


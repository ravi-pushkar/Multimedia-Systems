from articles import articles
from os import system, name
from tf_idf_search import TfIdfSearchEngine
from word2vec_search import Word2VecSearchEngine
  
import time

def clear_screen():
    if name == 'nt':
        _ = system('cls')
  
    else:
        _ = system('clear')
  

tf_idf_se = TfIdfSearchEngine()
word2vec_se = Word2VecSearchEngine()

load = False

if not load:
    tf_idf_se.build(articles)
    word2vec_se.build(articles)

    tf_idf_se.save()
    word2vec_se.save()

else:
    tf_idf_se.load()
    word2vec_se.load()


clear_screen()

while True:
    query = input("Please enter search query: ").strip()
    if query == 'exit':
        break
    print('Tf-Idf search results...')
    st = time.time()
    tf_idf_se.getSearchResults(query)
    print(f'Time taken {time.time() - st} s...')
    print('Word2Vec search results...')
    st = time.time()
    word2vec_se.getSearchResults(query)
    print(f'Time taken {time.time() - st} s...')


import re
from typing import Text
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from clean_list import *
import extract
import nltk


class TextCleaner():
    def __init__(self) -> None:
        nltk.download('omw-1.4')
        nltk.download('punkt')
        nltk.download('wordnet')

        self.ps = PorterStemmer()
        self.all_stop_words_base = [self.ps.stem(w) for w in all_stop_words]
        self.lemmatizer = WordNetLemmatizer()


    def expand_short_forms(self, word):
        if(word.lower() in contractions):
            return contractions[word.lower()]
        else:
            return word

        
    def cleanText(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\d'\s]+",'',text.lower())
        words = text.split() 
        base_words = [self.ps.stem(w) for w in words] 

        filtered_words = []
        for w in base_words:
            if w in self.all_stop_words_base:
                continue
            if w in contractions:
                w = contractions[w]

            w = self.lemmatizer.lemmatize(w)
            filtered_words.append(w)

        return " ".join(filtered_words)




# tc = TextCleaner()

# artc = extract.getText('TaylorSwift')

# print(tc.cleanText(artc))
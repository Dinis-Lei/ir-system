"""
Base template created by: Tiago Almeida & Sérgio Matos
Authors: Afonso Campos, Dinis Lei

Tokenizer module

Holds the code/logic addressing the Tokenizer class
and implemetns logic in how to process text into
tokens.

"""


import math
from utils import dynamically_init_class
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import re
import json
import time
import gc
from LFUCache import LFUCache


def dynamically_init_tokenizer(**kwargs):
    """Dynamically initializes a Tokenizer object from this
    module.

    Parameters
    ----------
    kwargs : Dict[str, object]
        python dictionary that holds the variables and their values
        that are used as arguments during the class initialization.
        Note that the variable `class` must be here and that it will
        not be passed as an initialization argument since it is removed
        from this dict.
    
    Returns
        ----------
        object
            python instance
    """
    return dynamically_init_class(__name__, **kwargs)

class Tokenizer:
    """
    Top-level Tokenizer class
    
    This loosly defines a class over the concept of 
    an index.

    """
    def __init__(self, **kwargs):
        super().__init__()
    
    def tokenize(self, text):
        """
        Tokenizes a piece of text, this should be
        implemented by specific Tokenizer sub-classes.
        
        Parameters
        ----------
        text : str
            Sequence of text to be tokenized
            
        Returns
        ----------
        object
            An object that represent the output of the
            tokenization, yet to be defined by the students
        """
        raise NotImplementedError()

        
class PubMedTokenizer(Tokenizer):
    """
    An example of subclass that represents
    a special tokenizer responsible for the
    tokenization of articles from the PubMed.

    """
    def __init__(self, 
                 minL, 
                 stopwords_path, 
                 stemmer,
                 *args, 
                 **kwargs):
        
        super().__init__(**kwargs)
        self.minL = minL
        self.stopwords_path = stopwords_path
        self.stemmer_name = stemmer
        print("init PubMedTokenizer|", f"{minL=}, {stopwords_path=}, {stemmer=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")
        # match any words of size bigger than min length
        self.pattern = re.compile(f'[a-zA-Z]\w{{{self.minL - 1},}}')
        self.stopwords = set()
        if self.stopwords_path:
            with open(self.stopwords_path, "r") as stopwords_file:
                self.stopwords = set(json.load(stopwords_file))

        self.stemmer = None
        if self.stemmer_name:
            if self.stemmer_name == "potterNLTK":
                self.stemmer = PorterStemmer()
            elif self.stemmer_name == "snowballNLTK":
                self.stemmer = SnowballStemmer(language='english')

        self.hasCache = False
        if kwargs.get('cache', None):
            self.hasCache = True
            self.cache = LFUCache(kwargs.get('cache'))
        

    def get_args(self):
        return {"class": "PubMedTokenizer", "minL": self.minL, "stopwords_path": self.stopwords_path, "stemmer": self.stemmer_name}

    def tokenize(self, document: str):
        token_stream = []
        idx = 0
        for word in re.findall(self.pattern, document.rstrip("").lower()):
            word: str
            if word not in self.stopwords:
                if not self.stemmer:
                    token_stream.append((word,idx))
                else:
                    if self.hasCache:
                        val = self.cache.get(word)
                        if val == -1:
                            stemmed = self.stemmer.stem(word, True)
                            self.cache.set(word, stemmed)
                        else:
                            stemmed = val
                    else:
                        stemmed = self.stemmer.stem(word, True)
                    token_stream.append((stemmed,idx))
            idx += 1
        return token_stream

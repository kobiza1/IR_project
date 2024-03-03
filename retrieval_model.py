import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.corpus import wordnet

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

STOP_WORDS = frozenset(stopwords.words("english"))
CORPUS_STOP_WORDS = ["category", "references", "also", "links", "external", "people"]


class retrievalModel:

    def __init__(self):
        self.BM25_dic = dict()
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet

p_stemmer = PorterStemmer()
def stem_post(post):
    split_post = post.split(' ')
    return ' '.join([p_stemmer.stem(word) for word in split_post])
#cite 6/9 Breakfast Hour
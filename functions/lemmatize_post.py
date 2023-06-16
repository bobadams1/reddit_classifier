import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet

lemmatizer = WordNetLemmatizer()
# cite: Lesson 504 NLP 1 - Modified to handle complete words.
def lemmatize_post(post):
    mapper = { 
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    post_split = post.split(' ')
    post_tokens = [(token, tag) for token, tag in nltk.pos_tag(post_split)]
    post_lem = []
    for token in post_tokens:
        pos = mapper.get(token[1][0])
        # post_lem.append((token[0],pos) if pos != None else (token[0]))
        post_lem.append(lemmatizer.lemmatize(token[0], pos) if pos != None else token[0])
    return ' '.join(post_lem).lower()
import re
import Config.text
import contractions
import nltk

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
from TextPreprocessing.Helpers import Smileys
from TextPreprocessing.Helpers import Emojis
from TextPreprocessing.Helpers import SpellCheck
from TextPreprocessing.Helpers.StopWords import StopWords


def preprocess(text):
    downloadNLTK()

    text = remove_html_tags(text)
    text = replace_url(text)
    text = replace_atUser(text)
    text = replace_smiley(text)
    text = replace_emojis(text)
    text = remove_leetspeak(text)
    text = remove_numbers(text)
    text = check_spelling(text)
    text = replace_contractions(text)
    text = remove_punct(text)
    text = to_lower(text)
    text = lemmatize(text)
    text = clean_white_space(text)

    return text


def downloadNLTK():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet.zip')
        nltk.data.find('corpora/omw-1.4.zip')
    except KeyError:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')


def remove_html_tags(text):
    """
    take string input and clean string without tags.
    use regex to remove the html tags.
    """
    return re.sub('<[^<]+?>', '', text)


def remove_leetspeak(text):
    return re.sub(r"[A-Za-z]+\d+[A-Za-z]+|\d+[A-Za-z]+\d+|[A-Za-z]+\d+|\d+[A-Za-z]+", '', text).strip()


def replace_contractions(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))

    return ' '.join(expanded_words)


def replace_atUser(text):
    text = re.sub('@[^\s]+', 'atUser', text)
    return text


def replace_url(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'url', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text


def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text


def to_lower(text):
    return text.lower()


def remove_punct(text):
    return ''.join(c for c in text if c not in punctuation)


def stem(text):
    stemmer = SnowballStemmer(Config.text.english)

    stemmed_word = [stemmer.stem(word) for sent in nltk.sent_tokenize(text) for word in
                    nltk.word_tokenize(sent)]
    return " ".join(stemmed_word)


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()

    lemmatized_word = [lemmatizer.lemmatize(word) for sent in nltk.sent_tokenize(text) for word in
                       nltk.word_tokenize(sent)]
    return " ".join(lemmatized_word)


def remove_smiley(text):
    return Smileys.remove_smiley(text)


def replace_smiley(text):
    return Smileys.replace_smiley(text)


def remove_emojis(text):
    return Emojis.remove_emojis(text)


def replace_emojis(text):
    return Emojis.replace_emojis(text)


def remove_stopwords(text):
    return StopWords(text).remove_stopwords()


def clean_white_space(text):
    return re.sub(' +', ' ', text)


def check_spelling(text):
    return SpellCheck.check_spelling(text, lang='en')
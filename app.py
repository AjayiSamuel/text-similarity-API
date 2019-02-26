import json
from flask import Flask, request
import nltk
# Imports
import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
import string

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

app = Flask(__name__)


@app.route('/', methods=['GET'])
def default():
    return "Text similiarity comparision"


@app.route('/webhook', methods=['POST'])
def webhook():
    the_request = request.get_json(silent=True, force=True)
    print(json.dumps(the_request, indent=2))
    acceptedAnswer = the_request.get('acceptedAnswer')
    print("The accepted answer is", acceptedAnswer)
    givenAnswer = the_request.get('givenAnswer')
    print("The given answer is", givenAnswer)

    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)
    stopwords.append('')

    def get_wordnet_pos(pos_tag):
        if pos_tag[1].startswith('J'):
            return (pos_tag[0], wordnet.ADJ)
        elif pos_tag[1].startswith('V'):
            return (pos_tag[0], wordnet.VERB)
        elif pos_tag[1].startswith('N'):
            return (pos_tag[0], wordnet.NOUN)
        elif pos_tag[1].startswith('R'):
            return (pos_tag[0], wordnet.ADV)
        else:
            return (pos_tag[0], wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()

    def is_ci_partial_noun_set_token_stopword_lemma_match(a, b):
        """Check if a and b are matches."""
        pos_a = map(get_wordnet_pos, nltk.pos_tag(word_tokenize(a)))
        pos_b = map(get_wordnet_pos, nltk.pos_tag(word_tokenize(b)))
        lemmae_a = [lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_a \
                    if pos == wordnet.NOUN and token.lower().strip(string.punctuation) not in stopwords]
        lemmae_b = [lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_b \
                    if pos == wordnet.NOUN and token.lower().strip(string.punctuation) not in stopwords]

        # Calculate Jaccard similarity
        ratio = len(set(lemmae_a).intersection(lemmae_b)) / float(len(set(lemmae_a).union(lemmae_b)))
        return (ratio > 0.66)

    result = is_ci_partial_noun_set_token_stopword_lemma_match(acceptedAnswer, givenAnswer)

    if result is True:
        response = "Correct"
    elif result is False:
        response = "Incorrect"
    else:
        response = "Your accessment is undecided"

    print(response)
    return response


if __name__ == '__main__':
    app.run(port=8080, debug=False)


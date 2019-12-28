import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

import collections
import math
import sys

from tabulate import tabulate

def init_nltk():
    try:
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
    except:
        pass


class WayneId:
    def __init__(self, text, targets):
        init_nltk()
        self.lemmatizer = WordNetLemmatizer()

        self.text = text
        self.targets = targets

        self.lemmatized_sentences = []
        self.context_words = set()
        # POS Tag dictionary
        self.tag_dict = dict()
        # target vs context vector
        self.vector = dict()

        self.corpus = text.lower()

        # lemmatize, build tags
        self.lem_sentences()
        self.build_tag_dict()

    # lemmatizes a single sentence
    def lemmatize_sentence(self, sentence):
        tagged = pos_tag(word_tokenize(sentence))
        lemmas = []

        for word, tag in tagged:
            # since we're using WordnetLemmatizer
            # wn stands for wordnet
            wntag = tag[0].lower()
            """
                in wordnet definition,
                
                a = adjective
                n = noun
                v = verb
                r = adverb
            """
            wntag = wntag if wntag in ['a', 'n', 'v', 'r'] else None
            if not wntag:
                lemmas.append(word)
            else:
                lemmas.append(self.lemmatizer.lemmatize(word, wntag))

        return ' '.join(lemmas)

    # lemmatizes all sentences
    def lem_sentences(self):
        self.lemmatized_sentences = []
        sentences = self.corpus.split('.')

        for sentence in sentences:
            self.lemmatized_sentences.append(self.lemmatize_sentence(sentence))

    # builds a POS dict for all the words in the text
    def build_tag_dict(self):
        # build pos tag dictionary
        for sentence in self.lemmatized_sentences:
            tagged = pos_tag(word_tokenize(sentence))

            for word, tag in tagged:
                self.tag_dict[word] = tag

    # only valid context to be taken here are adjectives, nouns and verbs
    def is_valid_context_word(self, word):
        return self.tag_dict[word][0].lower() in ['j', 'n', 'v']

    # picks valid context words from a sub array
    def process_token_subarray(self, sub, word_window):
        ret = []
        i = 0
        for s in sub:
            if self.is_valid_context_word(s) and i <= word_window:
                ret.append(s)
                i = i + 1
        return ret

    # populates context_words with valid words
    def process_list(self, word_list):
        for word in word_list:
            self.context_words.add(word)

    # builds the target vs context vector
    def build_vector(self, word_window):
        # init
        target_v_context = {}
        for target in self.targets:
            target_v_context[target] = []

        # process
        for sentence in self.lemmatized_sentences:
            tokens = sentence.split(' ')
            for token in tokens:
                if token in self.targets:
                    self.context_words.add(token)

                    i = tokens.index(token)

                    # split the list into left and right and then process them
                    left = tokens[:i]
                    right = tokens[i:]

                    processed_left = self.process_token_subarray(left, word_window)
                    processed_right = self.process_token_subarray(right, word_window)

                    self.process_list(processed_left + processed_right)
                    # update the vector
                    target_v_context[token] = target_v_context[token] + processed_left + processed_right

        # clean up self matches
        for target in target_v_context.keys():
            for cw in target_v_context[target]:
                if cw == target:
                    i = target_v_context[target].index(cw)
                    target_v_context[target].pop(i)

        features = self.targets + ['batman * wayne', 'joker * wayne']

        for cw in self.context_words:
            self.vector[cw] = {}
            for f in features:
                self.vector[cw][f] = 0

        for target in target_v_context.keys():
            c = collections.Counter(target_v_context[target])

            for v in self.vector.keys():
                self.vector[v][target] = c[v]

    # computes cosine similarity
    def cosine_sim(self, uv, u, v):
        return uv / (math.sqrt(u) * math.sqrt(v))

    # returns a tuple containing output, cosine sims and word_window
    def identify_wayne(self, word_window):
        # features for contexts
        features = self.targets + ['batman * wayne', 'joker * wayne']
        for cw in self.context_words:
            self.vector[cw] = {}
            for f in features:
                self.vector[cw][f] = 0

        # build vector
        self.build_vector(word_window=word_window)

        # dot product
        for v in self.vector.keys():
            self.vector[v]['batman * wayne'] = self.vector[v]['batman'] * self.vector[v]['wayne']
            self.vector[v]['joker * wayne'] = self.vector[v]['joker'] * self.vector[v]['wayne']

        # count features
        dim_features = {}
        for f in features:
            dim_features[f] = 0

        for v in self.vector.keys():
            for d in dim_features.keys():
                dim_features[d] = dim_features[d] + self.vector[v][d]

        # cosine similarity
        w_bat = self.cosine_sim(dim_features['batman * wayne'], dim_features['batman'], dim_features['wayne'])
        w_joker = self.cosine_sim(dim_features['joker * wayne'], dim_features['joker'], dim_features['wayne'])

        result = 'Wayne is Batman' if (1 - w_bat) < (1 - w_joker) else 'Wayne is Joker!'

        return [result, w_bat, w_joker, word_window]


# data & init
text = 'Batman is an American superhero.  The Joker is a supervillain that embodies the ideas of anarchy and chaos. ' \
       'The Joker and Batman fight the battle for Gotham’s soul. Like them, Bruce Wayne, an American billionaire is ' \
       'also from Gotham City. Is he hiding behind the face of these two? '
targets = ['batman', 'wayne', 'joker']
wayne_id = WayneId(text=text, targets=targets)

# run
word_window = int(sys.argv[1])
result = wayne_id.identify_wayne(word_window=word_window)
print()
print(tabulate([result], headers=['result', 'sim(batman, wayne)', 'sim(joker, wayne)', 'window_size'], tablefmt='orgtbl'))


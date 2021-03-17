import preprocessing as pp
import nltk
nltk.download("punkt")
import numpy as np
import random
import ast
import os
import logging
from operator import itemgetter
import pymongo
import pickle
import pandas as pd
from datetime import datetime, timedelta


class Similarity:
    client = pymongo.MongoClient(
        "mongodb+srv://cohitai:malzeit1984@cluster0.ufrty.mongodb.net/Livingdocs?retryWrites=true&w=majority")

    def __init__(self, model):
        self.word_vectors = model.wv
        self.df = None
        self.L2M = self.client.Livingdocs.articles_sqlike

    @staticmethod
    def cosine_similarity(u, v):

        """cosine similarity reflects the degree of similarity between u and v

        Arguments:
            u -- a word vector of shape (n,)
            v -- a word vector of shape (n,)

        Returns:
            cosine_similarity -- the cosine similarity
            between u and v defined by the formula above."""

        # Compute the dot product between u and v
        dot = np.dot(u, v)

        # Compute the L2 norm of u
        norm_u = np.sqrt(np.sum(u ** 2))

        # Compute the L2 norm of v
        norm_v = np.sqrt(np.sum(v ** 2))

        # Compute the cosine similarity
        if (norm_u * norm_v) == 0:
            return -1
        return dot / (norm_u * norm_v)

    def w2v_map(self, string):

        """function receives a string and returns its Word2Vec representation"""

        return np.array(self.word_vectors[string])

    def in_vocabulary(self, word):

        """function checks if a word exists in the vocabulary"""

        return word in self.word_vectors.vocab

    def sentence_to_avg(self, token_text):

        """function vectorizes text (list of strings) into the embedded space.
        Extracts the w2v representation of each word
        and then averages the list into a single
        vector.

        :param
        token_text: string representing a list of lists originated from an article.

        :returns
        average vector encoding information
        about the article, as numpy-array."""

        # unpacking the string back into a list of lists.
        try:
            token_list = ast.literal_eval(token_text)
        except ValueError:
            token_list = token_text
        averages_list = []
        number_of_sentences = len(token_list)

        for sentence in token_list:
            words = [self.w2v_map(w) for w in sentence if self.in_vocabulary(w)]

            if len(words) == 0:
                number_of_sentences -= 1
                continue

            # initialize the average word vector (same shape as word vectors).

            avg = np.mean(words, axis=0)
            averages_list.append(avg)

        sum_temp = np.zeros(self.w2v_map(random.choice(list(self.word_vectors.vocab))).shape)

        for v in averages_list:
            sum_temp += v

        # the case of an empty text which may occur when the article becomes empty after the prepossessing filtering.

        if number_of_sentences == 0:
            return sum_temp
        else:
            return sum_temp / number_of_sentences

    def add_average_vector(self):

        """method to compute and to add the average vector feature"""

        self.df["Average_vector"] = self.df["Tokenized_sents"].apply(self.sentence_to_avg)

    def find_similar_article(self, n, k):

        """function finds k similar articles to an article with index n;
        Argument: integers n,k.
                  a data frame df
        Returns : k integer indices similar to n."""

        logging.info("Find similar articles for article: {0}".format(n))

        list_distances = []
        for i in range(self.df.shape[0]):
            if i == n:
                continue
            list_distances.append(
                (i, self.cosine_similarity(self.df["Average_vector"][n], self.df["Average_vector"][i])))
        return sorted(list_distances, key=itemgetter(1))[-k:]

    def predict(self, k):
        """:param k: number of predictions
           :return a dictionary"""
        return {
            self.i2docid(i): [self.docid_url_dict()[self.i2docid(tup[0])] for tup in self.find_similar_article(i, k)][
                             ::-1] for i in range(self.df.shape[0])}

    def i2docid(self, i):
        return self.df.iloc[i]["id"]

    def docid_url_dict(self):
        return self.df.set_index('id').to_dict()['url']

    def create_test_df_sample(self, d, path_exclusion):

        # initiating an empty list if it does not exist yet.
        if not os.path.exists(path_exclusion+'ext.pkl'):
            with open(path_exclusion+'ext.pkl', 'wb') as f:
                pickle.dump([], f)

        logging.info("creating sample df...")
        delt = (datetime.now() - timedelta(days=d)).isoformat() + "Z"
        with open(path_exclusion + 'ext.pkl', 'rb') as f:
             ex_list = pickle.load(f)
        cur = self.L2M.find({"publishdate": {"$gte": delt}, "language": "German", "id": {"$nin": ex_list}})
        # cur = self.L2M.find({"publishdate": {"$gte": delt}, "language": "German"})
        df = pd.DataFrame(list(cur))
        df["Full Text"] = df["title"] + ' ' + df["lead"] + ' ' + df["text"]
        df["Tokenized_sents"] = df["Full Text"].apply(nltk.sent_tokenize)
        df["Tokenized_sents"] = df["Tokenized_sents"].apply(pp.clean_text_from_text)
        self.df = df

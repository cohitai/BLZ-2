import requests
import logging
import time
import pickle
import LSH as localhash

class AutoServer:

    """server automation"""
    def __init__(self, server_name, model, similarity, path_prediction, path_exclusion):
        self.path_prediction = path_prediction
        self.path_exclusion = path_exclusion
        self.server_name = server_name
        self.model = model
        self.similarity = similarity

    def automate(self, s, t, p_bool=False, algo=False, days=30):
        logging.info("Starting automation:")
        cnt = 1
        while True:
            if not cnt % s:
                # fit a model:
                self.model.fit(500, 20, 10, 4)

            # model load
            self.similarity.word_vectors = self.model.model.wv

            # create a database.
            self.similarity.create_test_df_sample(days, self.path_exclusion)

            # create a json file for prediction
            if not algo:
                pickle.dump(self.similarity.predict(k=6), open(self.path_prediction + 'model.pkl', 'wb'))
            else:
                LSH = localhash.LSH(self.similarity.df, self.model.model)
                LSH.make_hush_tables()
                LSH.make_recommendations()
                pickle.dump(LSH.make_recommendations(), open(self.path_prediction + 'model.pkl', 'wb'))

            if p_bool:
                files = {'file': open(self.path_prediction + 'model.pkl', 'rb')}
                # post
                r = requests.post(self.server_name + "/uploader", files=files)
                logging.info(r.text)

            cnt += 1
            logging.info("going to sleep...")
            time.sleep(t)

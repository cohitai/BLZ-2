import numpy as np 
import logging
import math

# dev
#np.random.seed(0)

#####
# PARAMETERS #
#####


class LSH:

    """class to apply LSH search"""
    
    def __init__(self, df, model):
        self.word_vectors = model.wv
        self.df = df
        self.df_url = df.set_index('id').to_dict()['url']
        # mat is a matrix with first column docids
        # and second column word vectors (docids | wv ) . Its dimension is (sample size, 2)
        self.mat = np.hstack((self.df['id'].to_numpy()[:, np.newaxis], self.df['Average_vector'].to_numpy()[:, np.newaxis]))
        self.number_of_buckets = int(self.mat.shape[0]/8)
        self.N_UNIVERSES = 25
        self.N_DIMS = len(self.mat[0][1])
        self.N_PLANES = math.ceil(np.log2(self.number_of_buckets))
        self.planes_l = [np.random.normal(size=(self.N_DIMS, self.N_PLANES)) for _ in range(self.N_UNIVERSES)]

        self.hash_tables = []
        self.id_tables = []

        logging.info('LSH INFO:')
        logging.info('Number of buckets:{}'.format(self.number_of_buckets))
        logging.info('Number of planes:{}'.format(self.N_PLANES))
        logging.info('Number of universes:{}'.format(self.N_UNIVERSES))
        logging.info('Number of dimensions:{}'.format(self.N_DIMS))


    @staticmethod
    def hash_value_of_vector(v, planes):
        """Create a hash for a vector; hash_id says which random hash to use.
        Input:
            - v:  vector of an article. It's dimension is (1, N_DIMS)
            - planes: matrix of dimension (N_DIMS, N_PLANES) - the set of planes that divide up the region
        Output:
            - res: a number which is used as a hash for your vector

        """
        # for the set of planes,
        # calculate the dot product between the vector and the matrix containing the planes.
        # planes has shape (300, 10)
        # The dot product will have the shape (1,10)
        dot_product = np.dot(v, planes)

        # get the sign of the dot product (1,10) shaped vector
        sign_of_dot_product = np.sign(dot_product)

        # set h to be false (equivalent to 0 when used in operations) if the sign is negative,
        # and true (equivalent to 1) if the sign is positive (1,10) shaped vector
        h = (sign_of_dot_product+1)/2

        # remove extra un-used dimensions (convert this from a 2D to a 1D array)
        h = np.squeeze(h)

        # initialize the hash value to 0
        hash_value = 0

        n_planes = planes.shape[1]
        for i in range(n_planes):
            # increment the hash value by 2^i * h_i
            hash_value += np.dot(np.power(2, i), h[i])

        # cast hash_value as an integer
        hash_value = int(hash_value)

        return hash_value

    def make_hash_table(self, vecs, planes):

        # number of planes is the number of columns in the planes matrix
        num_of_planes = planes.shape[1]

        # number of buckets is 2^(number of planes)
        num_buckets = np.power(2, num_of_planes)

        # create the hash table as a dictionary.
        # Keys are integers (0,1,2.. number of buckets)
        # Values are empty lists
        hash_table = {i: [] for i in range(num_buckets)}

        # create the id table as a dictionary.
        # Keys are integers (0,1,2... number of buckets)
        # Values are empty lists
        id_table = {i: [] for i in range(num_buckets)}

        for i in range(vecs.shape[0]):
            hash_value = self.hash_value_of_vector(vecs[i][1], planes)
            doc_id = vecs[i][0]

            # store the vector into hash_table at key h,
            # by appending the vector v to the list at key h
            hash_table[hash_value].append(vecs[i][1])

            # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
            # the key is the h, and the 'i' is appended to the list at key h
            id_table[hash_value].append(doc_id)
        return hash_table, id_table

    def make_hush_tables(self):

        # create hush tables

        self.hash_tables = []
        self.id_tables = []

        for universe_id in range(self.N_UNIVERSES):
            logging.info('working on hash universe #: {}'.format(universe_id))
            planes = self.planes_l[universe_id]
            hash_table, id_table = self.make_hash_table(self.mat, planes)
            self.hash_tables.append(hash_table)
            self.id_tables.append(id_table)

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

    def nearest_neighbor(self, v, candidates, k=1):
        """
        Input:
        - v, the vector you are going find the nearest neighbor for
        - candidates: a set of vectors where we will find the neighbors
        - k: top k nearest neighbors to find
        Output:
        - k_idx: the indices of the top k closest vectors in sorted form min to max.
        """
        similarity_l = []

        # for each candidate vector...
        for row in candidates:
            # get the cosine similarity
            cos_similarity = self.cosine_similarity(v, row)

            # append the similarity to the list
            similarity_l.append(cos_similarity)
        
        # sort the similarity list and get the indices of the sorted list
        sorted_ids = np.argsort(similarity_l)

        # get the indices of the k most similar candidate vectors
        k_idx = sorted_ids[-k:]
        return k_idx

    def approximate_knn(self, doc_id, v, planes_l, k, num_universes_to_use):

        """Search for k-NN using hashes."""
        assert num_universes_to_use <= self.N_UNIVERSES

        # Vectors that will be checked as possible nearest neighbor
        vecs_to_consider_l = list()

        # list of document IDs
        ids_to_consider_l = list()

        # create a set for ids to consider, for faster checking if a document ID already exists in the set
        ids_to_consider_set = set()

        # loop through the universes of planes
        for universe_id in range(num_universes_to_use):

            # get the set of planes from the planes_l list, for this particular universe_id
            planes = planes_l[universe_id]

            # get the hash value of the vector for this set of planes
            hash_value = self.hash_value_of_vector(v, planes)

            # get the hash table for this particular universe_id
            hash_table = self.hash_tables[universe_id]

            # get the list of document vectors for this hash table, where the key is the hash_value
            document_vectors_l = hash_table[hash_value]

            # get the id_table for this particular universe_id
            id_table = self.id_tables[universe_id]

            # get the subset of documents to consider as nearest neighbors from this id_table dictionary
            new_ids_to_consider = id_table[hash_value].copy()

            # remove the id of the document that we're searching
            if doc_id in new_ids_to_consider:

                new_ids_to_consider.remove(doc_id)
                logging.info("removed doc_id {} of input vector from new_ids_to_search".format(doc_id))

            # loop through the subset of document vectors to consider
            for i, new_id in enumerate(new_ids_to_consider):

                # if the document ID is not yet in the set ids_to_consider...
                if new_id not in ids_to_consider_set:
                    # access document_vectors_l list at index i to get the embedding
                    # then append it to the list of vectors to consider as possible nearest neighbors
                    document_vector_at_i = document_vectors_l[i]
                    vecs_to_consider_l.append(document_vector_at_i)

                    # append the new_id (the index for the document) to the list of ids to consider
                    ids_to_consider_l.append(new_id)

                    # also add the new_id to the set of ids to consider
                    # (use this to check if new_id is not already in the IDs to consider)
                    ids_to_consider_set.add(new_id)

        # Now run k-NN on the smaller set of vecs-to-consider.
        logging.info("Fast considering {} vecs".format(len(vecs_to_consider_l)))

        # convert the vecs to consider set to a list, then to a numpy array
        vecs_to_consider_arr = np.array(vecs_to_consider_l)

        # call nearest neighbors on the reduced list of candidate vectors
        nearest_neighbor_idx_l = self.nearest_neighbor(v, vecs_to_consider_arr, k=k)

        # Use the nearest neighbor index list as indices into the ids to consider
        # create a list of nearest neighbors by the document ids
        nearest_neighbor_ids = [ids_to_consider_l[idx]
                                for idx in nearest_neighbor_idx_l]

        return nearest_neighbor_ids[::-1]

    def make_recommendations(self):
        return {self.mat[i][0]: [self.df_url[x] for x in self.approximate_knn(
            self.mat[i][0], self.mat[i][1], self.planes_l, k=6, num_universes_to_use=self.N_UNIVERSES)] for i in range(self.mat.shape[0])}

        
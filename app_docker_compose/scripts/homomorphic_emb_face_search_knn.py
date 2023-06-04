"""
Search and retrieval with KNN for homomorphically encryted face embeddings
Only Client holds the private key to decrypt the embeddings. 
The Server can however store the encrypted embeddings and calculate the squared euclidean distance between existing
encrypted embeddings.

For full client-server example: 
    https://pyfhel.readthedocs.io/en/latest/_autoexamples/Demo_5_CS_Client.html#sphx-glr-autoexamples-demo-5-cs-client-py

requirements:
    pip install Pyfhel
"""
from typing import List

import numpy as np
from Pyfhel import Pyfhel, PyCtxt

EMB_LENGTH = 128


def get_pyfhel_obj():
    """
    Returns a Pyfhel encryption/decryption object
    """
    HE = Pyfhel()           # Creating empty Pyfhel object
    ckks_params = {
        'scheme': 'CKKS',   # can also be 'ckks'
        'n': 2**14,         # Polynomial modulus degree. For CKKS, n/2 values can be
                            #  encoded in a single ciphertext.
                            #  Typ. 2^D for D in [10, 15]
        'scale': 2**30,     # All the encodings will use it for float->fixed point
                            #  conversion: x_fix = round(x_float * scale)
                            #  You can use this as default scale or use a different
                            #  scale on each operation (set in HE.encryptFrac)
        # Number of bits of each prime in the chain.
        'qi_sizes': [60, 30, 30, 30, 60]
        # Intermediate values should be  close to log2(scale)
        # for each operation, to have small rounding errors.
    }
    HE.contextGen(**ckks_params)  # Generate context for ckks scheme
    HE.keyGen()             # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()
    return HE


def l2_dist_sq_ctxt(vec_1: PyCtxt, vec_2: PyCtxt, HE: Pyfhel):
    """
    Calculates the encrypted squared l2 norm between encrypted vectors vec_1 and vec_2
    """
    dist_sq = ~(vec_1 - vec_2) ** 2
    dist_sq = HE.cumul_add(dist_sq)
    return dist_sq


class Client:
    """
    Pyfhel Client. Holds the secret key and encrypts vectors to send to the server
    """

    def __init__(self, context_params: dict) -> None:
        print("[Client] Initializing Pyfhel session and data...")
        self.he_client = Pyfhel(context_params=context_params)
        self.he_client.keyGen()  # gen both a public & a private key
        self.he_client.relinKeyGen()
        self.he_client.rotateKeyGen()
        # Serializing public context information
        self.public_context = {
            "s_context": self.he_client.to_bytes_context(),
            "s_public_key": self.he_client.to_bytes_public_key(),
            "s_relin_key": self.he_client.to_bytes_relin_key(),
            "s_rotate_key": self.he_client.to_bytes_rotate_key(),
        }

    def get_encrypted_vector_bytes(self, vector: np.ndarray):
        """Encrypts & returns vector obj"""
        ctx = self.he_client.encrypt(vector)

        # Serializing data
        s_ctx = ctx.to_bytes()
        return s_ctx


class Server:
    """
    Pyfhel Server. Holds the encrypted vector database. 
    Runs the first part of KNN to find all neighbors and send the encrypted neighbor distances back to the client,
    which sends the decryted the distances back to the server & 
    then the server responds with the nearest encrypted neighbor vectors
    """

    def __init__(self) -> None:
        print("[Server] Starting")
        self.he_server = Pyfhel()
        self.vector_db = []

    def connec_to_client(self, client_public_context: dict):
        """
        Connects to client HE public context. 
        client_public_context is a dict with attributes {s_context, s_public_key, s_relin_key, s_rotate_key}
        """
        # Read all bytestrings from client
        self.he_server.from_bytes_context(
            client_public_context["s_context"])
        self.he_server.from_bytes_public_key(
            client_public_context["s_public_key"])
        self.he_server.from_bytes_relin_key(
            client_public_context["s_relin_key"])
        self.he_server.from_bytes_rotate_key(
            client_public_context["s_rotate_key"])

    def register_vector(self, s_ctx: bytes):
        """
        Register encrypted vector into database.
        Server must be connected to client first
        """
        ctx = PyCtxt(pyfhel=self.he_server, bytestring=s_ctx)
        self.vector_db.append(ctx)

    def get_all_vector_distances(self, s_ctx: bytes):
        """
        Computes the squared l2-dist of vector repr by s_ctx to all vectors in existing vector db
        And returns a list of bytes of the encrypted vector distances
        Server must be connected to client first
        """
        if len(self.vector_db) == 0:
            raise ValueError("No encrypted vectors have been registered")
        rvector = PyCtxt(pyfhel=self.he_server, bytestring=s_ctx)

        vector_dists = []
        for vector in self.vector_db:
            dist = l2_dist_sq_ctxt(rvector, vector, self.he_server)
            vector_dists.append(dist.to_bytes())

        return vector_dists

    def get_knn(self, dist_list: List, k: int = 1):
        """
        Get the k nearest neighbors from the dist list
        i.e. take the min-k dist values and get the respective vectors in vector db
        """
        dist_list = [[dist, i] for i, dist in enumerate(dist_list)]
        dist_list.sort()
        closest_vectors = [self.vector_db[i].to_bytes() for _, i in dist_list[:k]]
        return closest_vectors


def test_funcs():
    """
    Tests basic Pyfhel functions
    """
    # create vector dataset
    dataset = [np.random.random(EMB_LENGTH) for _ in range(100)]
    dataset = np.asarray(dataset)

    # example of a multiplication operation on encrypted vectors
    HE = get_pyfhel_obj()
    ctxt = HE.encrypt(dataset[0])
    ctxt *= ctxt
    HE.relinKeyGen()  # relinearize after ctxt-ctxt mults
    dtxt = HE.decrypt(ctxt)[:EMB_LENGTH]  # len(arrs) < n filled with 0s
    print(np.allclose(dtxt, dataset[0] * dataset[0], atol=0.0001))

    # sub, square and cumulative sum example, i.e. square of l2 dist
    a, b = np.asarray([2.0, 4.0, 6.0]), np.asarray([1.0, 1.0, 2.0])
    ctxt1 = HE.encrypt(a)
    ctxt2 = HE.encrypt(b)
    # equi to  HE.cumul_add(~(ctxt1 - ctxt2)**2)
    ctxt3 = l2_dist_sq_ctxt(ctxt1, ctxt2, HE)
    print(np.allclose(HE.decrypt(ctxt3)[:3], np.sum((a - b)**2), atol=0.1))


if __name__ == "__main__":
    # Testing a server/client setup
    server = Server()
    client = Client({'scheme': 'ckks', 'n': 2**13,
                    'scale': 2**30, 'qi_sizes': [30]*5})
    server.connec_to_client(client.public_context)

    # create vector dataset
    dataset = [np.random.random(EMB_LENGTH) for _ in range(100)]
    dataset = np.asarray(dataset)

    # save vectors in server vector db
    for vec_data in dataset:
        client_s_ctx = client.get_encrypted_vector_bytes(vec_data)
        server.register_vector(client_s_ctx)

    # query vector and get the list of enc vec distances back from the server
    QUERY_IDX = 2
    query_vec = dataset[QUERY_IDX]
    query_client_s_ctx = client.get_encrypted_vector_bytes(query_vec)
    enc_vec_dists = server.get_all_vector_distances(query_client_s_ctx)

    # decode vector dists
    dec_vec_dists = [client.he_client.decrypt(PyCtxt(pyfhel=client.he_client, bytestring=enc_vec))[0]
                     for enc_vec in enc_vec_dists]

    closest_idx = np.argmin(dec_vec_dists)
    assert QUERY_IDX == closest_idx

    # retrieve the vector with knn
    knn_vec_bytes = server.get_knn(dec_vec_dists, 1)
    knn_vec_enc = [PyCtxt(pyfhel=client.he_client, bytestring=byte_vec) 
                   for byte_vec in knn_vec_bytes]
    print(knn_vec_enc)
    assert np.allclose(client.he_client.decrypt(knn_vec_enc[0])[:EMB_LENGTH], dataset[QUERY_IDX], atol=0.00001)

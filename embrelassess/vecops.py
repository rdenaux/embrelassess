import struct
import numpy as np
from functools import partial
import codecs
import os


def read_swivel_vecs(binFile, vocabulary, dims):
    """Reads vector embeddings from a Swivel's bin and vocab

    The .bin file cointains the vector embeddings, which we unpack.
    Checks if the tokens in the vocabulary are in the EN - ES mapping, and then
    creates a tuple containing the syncons and their embeddings.

    Parameters
    ----------
    binFile
      Binary file of the vector embeddings of the vocabulary.
    vocabulary
      File containing the tokens and words.
    dims
      Size of the embedding vectors.

    Returns
    -------
    tuple
      Contains two aligned vectors, one with syncons and the other one with
      their respective embeddings.
    """
    vec = []
    tokens = []
    fmt = struct.Struct('%df' % dims)
    line_sep = '\n\r' if os.name == 'nt' else '\n'
    with open(binFile, 'rb') as f:
        with codecs.open(vocabulary, 'r', encoding='utf-8') as vocab:
            for data, token in zip(iter(partial(f.read, dims*4), ''), vocab):
                token = token.strip(line_sep)
                tokens.append(token)
                vec.append(fmt.unpack(data))
    return np.asarray(tokens), np.asarray(vec)

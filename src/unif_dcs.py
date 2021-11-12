
import sys
import pathlib

import numpy as np

from .search_models.sentence_search_model import Sentence_SearchModel
from .search_models import models
from . import data_generator, helper

class UNIF_DCS(Sentence_SearchModel):

    def __init__(self, data_path, data_chunk_id=0):
        self.data_path = data_path

        # dataset info
        self.total_length = data_generator.DCS_NUM_ELEMENTS
        self.chunk_size = 600000   # 18223872  # 10000

        self.max_length = 90

        number_chunks = self.total_length / self.chunk_size - 1
        self.number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

        self.data_chunk_id = min(data_chunk_id, int(self.number_chunks))
        print("### Loading UNIF model with DCS chunk number " + str(data_chunk_id) + " [0," + str(number_chunks)+"]")

        self.vocab_code, self.vocab_desc = None, None
        self.inverse_vocab_code, self.inverse_vocab_desc = None, None

        self.embedding_size = 2048
        self.number_code_tokens = 10001
        self.number_desc_tokens = 10001
        self.code_length = 90
        self.desc_length = 90
        self.hinge_loss_margin = 0.40

    def get_vocabularies(self):
        return self.load_vocabulary(self.data_path + "vocab.tokens.pkl", self.data_path + "vocab.desc.pkl")

    def get_model(self):
        self.training_model, self.model_code, self.model_query, self.dot_model = models.unif_model(
                        self.embedding_size, self.number_code_tokens, self.number_desc_tokens,
                        self.code_length, self.desc_length, self.hinge_loss_margin)

    def desc_tokenizer(self, desc):
        tokenized = []

        for word in desc.split(" "):
            if word in self.inverse_vocab_desc:
                tokenized.append(self.inverse_vocab_desc[word])
            else:
                tokenized.append(self.inverse_vocab_desc["UNK"])

        return helper.pad(np.array(tokenized).reshape((1,-1)), self.max_length)

    def code_tokenizer(self, code):
        tokenized = []

        for word in code.split(" "):
            if word in self.inverse_vocab_code:
                tokenized.append(self.inverse_vocab_code[word])
            else:
                tokenized.append(self.inverse_vocab_code["UNK"])

        return helper.pad(np.array(tokenized).reshape((1,-1)), self.max_length)

    def load_dataset(self, desc_path, code_path, vocab_desc, vocab_code, batch_size=16):

        ds = data_generator.get_dcs_dataset(desc_path, code_path, vocab_desc, vocab_code, max_len=-1)

        ds = ds.map(data_generator.naive_format_map(self.desc_tokenizer, self.code_tokenizer))

        ds = ds.batch(batch_size, drop_remainder=True)

        return ds


if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    unif = UNIF_DCS(data_path)

    unif.get_model()

    vocab_code, vocab_desc = unif.get_vocabularies()

    BATCH_SIZE = 16

    ds = unif.load_dataset(data_path+"/train.desc.h5", data_path+"/train.tokens.h5", vocab_desc, vocab_code, BATCH_SIZE)

    steps_per_epoch = data_generator.DCS_NUM_ELEMENTS // BATCH_SIZE

    #unif.train(ds, script_path + "/../weights/unif-weights", epochs=1, steps_per_epoch=steps_per_epoch)

    unif.load_weights(script_path + "/../kth_w/unif_600000_90_dcs_weights")

    test_ds = unif.load_dataset(data_path+"/test.desc.h5", data_path+"/test.tokens.h5", vocab_desc, vocab_code, 500)

    unif.test(test_ds, script_path+"/../results/unif-results")





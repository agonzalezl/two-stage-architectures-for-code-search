
import os
from .. import helper

class CodeSearchManager:

    def __init__(self):
        self.training_model, self.code_model = None, None
        self.desc_model, self.dot_model = None, None

    def load_weights(self, path):
        if os.path.isfile(path + '.index'):
            self.training_model.load_weights(path)
            print("Weights loaded!")
        else:
            print("Warning! No weights loaded!")

    def get_top_n(self, n, results):
        count = 0
        for r in results:
            if results[r] < n:
                count += 1
        return count / len(results)

    def train(self, training_set, weights_path, epochs=1, batch_size=None, steps_per_epoch=None):
        self.training_model.fit(training_set, epochs=epochs, verbose=1, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
        self.training_model.save_weights(weights_path)
        print("Model saved!")


    def load_vocabulary(self, code_vocab_path, desc_vocab_path):
        self.inverse_vocab_code = helper.load_pickle(code_vocab_path)
        self.vocab_code = {y: x for x, y in self.inverse_vocab_code.items()}

        self.inverse_vocab_desc = helper.load_pickle(desc_vocab_path)
        self.vocab_desc = {y: x for x, y in self.inverse_vocab_desc.items()}

        return self.vocab_code, self.vocab_desc


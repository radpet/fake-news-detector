import pickle


class TokenizerSerializer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def serialize(self, name):
        with open(name, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(name):
        with open(name, 'rb') as f:
            return pickle.load(f)

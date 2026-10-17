import json

from .constants import SPECIAL_TOKENS, UNK_ID, UNK_TOKEN


class Encoding():
    def __init__(self, ids, tokens):
        self.ids = ids
        self.tokens = tokens

    def __len__(self):
        return len(self.ids)


class CharacterTokenizer():
    def __init__(self, dataset_path=None, vocab=None):
        if vocab is not None:
            self.vocab = vocab

        elif dataset_path is not None:
            unique_chars = set()

            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    unique_chars.update(line.strip())

            unique_chars = sorted(unique_chars)
            self.vocab = SPECIAL_TOKENS + unique_chars

        self.char_to_id = {char: id for id, char in enumerate(self.vocab)}
        self.id_to_char = {id: char for id, char in enumerate(self.vocab)}

    def encode(self, text):
        ids = [self.char_to_id.get(char, UNK_ID) for char in text]
        tokens = [self.id_to_char.get(id, UNK_TOKEN) for id in ids]
        return Encoding(ids, tokens)

    def encode_batch(self, sentences):
        return [self.encode(sentence) for sentence in sentences]

    def decode(self, ids):
        chars = []
        for id in ids:
            chars.append(self.id_to_char.get(id, UNK_TOKEN))
        return ''.join(chars)

    def get_vocab_size(self):
        return len(self.vocab)

    def save(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = list(data.keys())

        return cls(vocab=vocab)

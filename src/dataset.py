import torch
from torch.utils.data import Dataset


class LMDataset(Dataset):
    def __init__(self, tokens, context_length):
        super().__init__()

        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, id):
        start = id * self.context_length
        chunk = self.tokens[
            start: start + self.context_length + 1
        ]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y


if __name__ == "__main__":
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    context_length = 4

    dataset = LMDataset(tokens=tokens, context_length=context_length)
    x, y = dataset[0]

    print(f'size of tokens: {len(tokens)}')
    print(f'context length: {context_length}')
    print(f'size of dataset: {len(dataset)}')
    print(f'x - window 0: {x} of type {x.dtype}')
    print(f'y - window 0: {y} of type {y.dtype}')

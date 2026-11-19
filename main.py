# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: lt2326-2026-assignment-1 (3.12.16)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Part 1: The Tokenizers
#
# ### Pre-experiment Predictions
#
# **Which language do you expect to require the most tokens?**
#  - Turkish to require the most tokens.
#    - Turkish and English both use Latin-based alphabets, but Turkish has _"more productive word morphology"_.
#    - Individual Chinese characters are often words on their own.
#
# **How do you expect the larger BPE vocabulary to change the representation of each language?**
#  - English — full words or even common short phrases.
#  - Turkish — I think it will be the same as English.
#  - Chinese — common 2 or more character words and even idioms
#
# **What kinds of units do you expect BPE to learn in English, Turkish, and Chinese?**
#  - English — frequently used words (e.g. the, and, a, etc...) and common prefixes and suffixes (e.g. -ing, -er, re-, etc...)
#  - Turkish — I think it will be the same as English.
#  - Chinese — frequently used characters (e.g. 的，了，个) and common short words.
#
# **Which tokenizer do you expect to work best for language modeling?**
#  - I'm guessing the small-sized BPE tokenizer might work best for language modeling.
#    - Character-level tokenization will be too slow and inefficient, and making it hard for the model to manage the context.
#    - If the BPE vocabulary is too large, there will be a lot more parameters for the model.
#    - If the BPE vocabulary is small and compressed at the sweet spot, it is more efficient for the model.
#

# %% [markdown]
# ### Pre-experiment Observation
#
# I noticed that the Chinese corpus contains a mixture of traditional and simplified Chinese characters (e.g. 大學 and 大学). Both have the same meaning but would impact the frequencies since they are considered different tokens. This affects evaluations further down the pipeline.
#
# As such, I have decided to do a normalization by converting all the Chinese characters to simplified Chinese characters for the entire dataset.
#
# To run the normalization, run the following command: `uv run ./script/normalize_chinese.py`.
#

# %%
# #!uv run ./script/normalize_chinese.py

# %% [markdown]
# ### Imports

# %%
import json
import math
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import Markdown, display
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import (EOS_ID, LANGUAGES, LANGUAGES_LABEL_MAPPING, LOCAL_RUN,
                 RANDOM_SEED, SPECIAL_TOKENS, UNK_ID, UNK_TOKEN,
                 CharacterTokenizer, DecoderTransformer, LMDataset)

# %% [markdown]
# ### Set the Device

# %%
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

is_cuda = torch.cuda.is_available()
is_mps = torch.backends.mps.is_available()

if is_cuda:
    DEVICE = torch.device("cuda")
elif is_mps:
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# %% [markdown]
# ### Character-level Tokenizer

# %%
tokenizer_directory = './artifact/tokenizer'

os.makedirs(tokenizer_directory, exist_ok=True)


character_tokenizer_file = f'{tokenizer_directory}/character-tokenizer.json'

if os.path.exists(character_tokenizer_file):
    print(f"Tokenizer already exists at {character_tokenizer_file}.")
    character_tokenizer = CharacterTokenizer.from_file(
        character_tokenizer_file
    )
else:
    character_tokenizer = CharacterTokenizer('./data/tokenizer/balanced.txt')
    character_tokenizer.save(character_tokenizer_file)

print(f"Total vocab size: {len(character_tokenizer.vocab):,d}")
print(character_tokenizer.vocab[:10])

print(
    character_tokenizer.encode('hello').ids,
    character_tokenizer.encode('hello').tokens,
    character_tokenizer.decode(character_tokenizer.encode('hello').ids)
)
print(character_tokenizer.encode("✨").ids, UNK_ID)


# %% [markdown]
# ### BPE Tokenizer

# %%
# # %%time
def train_bpe(vocab_size, input_file, output_file):
    if os.path.exists(output_file):
        print(f"Tokenizer already exists at {output_file}. Skip training.")
        return Tokenizer.from_file(output_file)

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    trainer = BpeTrainer(
        special_tokens=SPECIAL_TOKENS,
        vocab_size=vocab_size,
        show_progress=False
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train([input_file], trainer)
    tokenizer.save(output_file)
    return tokenizer


# %%
bpe_tokenizer_small = train_bpe(
    10000, './data/tokenizer/balanced.txt', f'{tokenizer_directory}/bpe-tokenizer-small.json')
bpe_tokenizer_big = train_bpe(
    38500, './data/tokenizer/balanced.txt', f'{tokenizer_directory}/bpe-tokenizer-big.json')

print(f"Small tokenizer's size: {bpe_tokenizer_small.get_vocab_size():,d}")
print(f"Big tokenizer's size: {bpe_tokenizer_big.get_vocab_size():,d}")


# %% [markdown]
# #### Reason for BPE Vocabulary Sizes
#
# I have chosen the vocabulary sizes of 10,000 for the small and 38,500 for the big.
#
# ##### Why 10,000 for small?
#
# The training corpus contains $7,648$ unique characters across all languages. Therefore, the minimum vocabulary size is $7,648$. I have chosen 10,000 so that a modest amount of merges ($10,000 - 7,648 = 2,352$) can be performed on the most frequent sub-words.
#
# ##### Why 38,500 for big?
#
# I take $7,648 \times 5$ which is $\approx 38,500$. This allows for a large amount of merges ($38,500 - 7,648 = 30,852$) to be performed, which will result in full words being merged as tokens.

# %% [markdown]
# #### Analyze the Tokenizers

# %%
def load_dataset(type):
    dataset = {}
    random.seed(RANDOM_SEED)
    sample_size = 1000 if LOCAL_RUN else None

    for language in LANGUAGES:
        with open(f'./data/{type}/{language}.txt', 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]

            if LOCAL_RUN:
                lines = random.sample(lines, min(len(lines), sample_size))

            dataset[language] = lines

    return dataset


valid_dataset = load_dataset("valid")

tokenizers = {
    "Character Tokenizer": character_tokenizer,
    "BPE Tokenizer (small)": bpe_tokenizer_small,
    "BPE Tokenizer (big)": bpe_tokenizer_big,
}

data = []
for tokenizer_name, tokenizer in tokenizers.items():
    vocab_size = tokenizer.get_vocab_size()

    for language, lines in valid_dataset.items():
        encoded = tokenizer.encode_batch(lines)
        all_token_ids = [
            token_id for encoding in encoded for token_id in encoding.ids
        ]

        total_tokens = len(all_token_ids)
        unique_tokens = len(set(all_token_ids))
        total_characters = sum(len(line) for line in lines)
        total_unk = sum(1 for token_id in all_token_ids if token_id == UNK_ID)

        data.append({
            "Tokenizer": tokenizer_name,
            "Language": LANGUAGES_LABEL_MAPPING[language],
            "Vocabulary Size": vocab_size,
            "Total Tokens": total_tokens,
            "Unique Tokens": unique_tokens,
            "Tokens/Sentence": total_tokens / len(lines),
            "Characters/Token": total_characters / total_tokens,
            "Total Characters": total_characters,
            "Total UNK": total_unk,
            "UNK Rate": total_unk/total_tokens,
        })

df = pd.DataFrame(data)
df.to_json(
    f"{tokenizer_directory}/tokenizer_metrics.json",
    orient="records",
    indent=2
)

df.style.format({
    "Vocabulary Size": "{:,d}",
    "Total Tokens": "{:,d}",
    "Unique Tokens": "{:,d}",
    "Tokens/Sentence": "{:.2f}",
    "Characters/Token": "{:.2f}",
    "Total Characters": "{:,d}",
    "Total UNK": "{:,d}",
    "UNK Rate": "{:.2%}",
})

# %% [markdown]
# ##### Observations
#
# - With BPE tokenizer, both English and Turkish merge about 3 to 4 characters into a token, while Chinese stays almost as single character per token. This makes sense as Chinese characters are already meaningful words by themselves and they don't require white spaces to separate them.
# - With character tokenizer, both English and Turkish have a high average of 80 to 100 tokens per sentence. But with BPE tokenizer, these numbers dropped to approximately 20 tokens per sentence. This allows models to process more information in a single step.
# - The UNK rate is very low for all languages across all tokenizers. This indicates that the training data is very sufficient for vocabulary coverage.

# %%
random.seed(RANDOM_SEED)

samples = {
    language: random.choice(valid_dataset[language]) for language in LANGUAGES
}

for language, sample in samples.items():
    print('=' * 10, f'{LANGUAGES_LABEL_MAPPING[language].upper()}', '=' * 10)
    print(sample)
    print(f"Character Tokenizer:\n{character_tokenizer.encode(sample).tokens}")
    print(
        f"BPE Tokenizer (small):\n{bpe_tokenizer_small.encode(sample).tokens}"
    )
    print(f"BPE Tokenizer (big):\n{bpe_tokenizer_big.encode(sample).tokens}")
    print('\n', '-' * 50, '\n')

# %% [markdown]
# ##### English
#
# In BPE Tokenizer (small), we can see that because of the limited vocabulary size, not enough merges to cater for most of the words. For example, `hotel` is tokenized into `h`, `ot` and `el`. However, this is not the case in BPE Tokenizer (big) where `hotel` is tokenized as a whole. This is because the vocabulary size is large enough to cater for most of the words.
#
# ##### Turkish
#
# Same observations as per English.
#
# ##### Chinese
#
# For both small and big versions of the BPE Tokenizers, we can see that the Chinese characters merge at 2 to 3 characters, which is again reflected in the metrics (low characters/token) above. This is because Chinese characters are atomic and there are a huge number of them. Common characters that appear together aren't high frequency enough compared to English and Turkish for the BPE algorithm to merge them.

# %% [markdown]
# ## Part 2: The Language Models

# %% [markdown]
# ### Prepare the data loaders

# %%
train_dataset = {}
for language in LANGUAGES:
    with open(f'./data/train/{language}.txt', 'r', encoding='utf-8') as f:
        train_dataset[language] = [line.strip() for line in f]


def create_data_loader(dataset, tokenizer, type='train', shuffle=True, context_length=128, batch_size=32):
    type_label = 'validation' if type == 'valid' else f'{type}ing'
    is_train = type == 'train'

    data = []

    for sentences in dataset.values():
        data.extend(sentences)

    # for language in LANGUAGES:
    #     with open(f'./data/{type}/{language}.txt', 'r', encoding='utf-8') as f:
    #         data.extend([line.strip() for line in f])

    print(
        f"Total {type_label} sentences across {len(dataset)} languages: {len(data):,d}")
    # print(
    #     f"Total {type_label} sentences across {len(LANGUAGES)} languages: {len(data):,d}"
    # )

    if is_train:
        random.seed(RANDOM_SEED)
        random.shuffle(data)

    print(f"Sample {type_label} data: {data[:10]}")

    encodings = tokenizer.encode_batch(data)
    # print(encodings[0], encodings[0].ids, encodings[0].tokens)
    token_ids = []

    for encoding in encodings:
        token_ids.extend(encoding.ids)
        token_ids.append(EOS_ID)

    print(f"Total {type_label} tokens: {len(token_ids):,d}")

    # print(
    # f"Total {type_label} tokens across {len(LANGUAGES)} languages: {len(token_ids):,d}")

    dataset = LMDataset(
        tokens=torch.tensor(token_ids),
        context_length=context_length
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train
    )

    return data_loader


def get_loaders(label, tokenizer):
    print("=" * 50)
    print("#" * 10, label.upper(), "#" * 10)
    print("-" * 50)
    train_loader = create_data_loader(train_dataset, tokenizer, type="train")
    print()
    valid_loader = create_data_loader(valid_dataset, tokenizer, type="valid")
    print("-" * 50)
    print("=" * 50)
    print()
    return train_loader, valid_loader


# %%
training_data_loader_char, validation_data_loader_char = get_loaders(
    "Character Tokenizer", character_tokenizer
)
training_data_loader_bpe_small, validation_data_loader_bpe_small = get_loaders(
    "BPE Tokenizer (small)", bpe_tokenizer_small
)
training_data_loader_bpe_big, validation_data_loader_bpe_big = get_loaders(
    "BPE Tokenizer (big)", bpe_tokenizer_big
)


# %% [markdown]
# ### Training and validation loop

# %%
def train_and_validate(tokenizer, training_data_loader, validation_data_loader):
    model = DecoderTransformer(vocabulary_size=tokenizer.get_vocab_size())
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=0.0001)

    training_batches = []
    training_losses = []
    validation_batches = []
    validation_losses = []

    validation_interval = len(training_data_loader) // 4

    NUM_EPOCH = 1

    for epoch in range(NUM_EPOCH):
        model.train()

        running_training_loss = 0.0

        training_progress_bar = tqdm(
            training_data_loader,
            f"[Training] Epoch {epoch + 1} / {NUM_EPOCH}"
        )

        for batch, (x, y) in enumerate(training_progress_bar, 1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            # print(logits.shape, logits.size(2))

            logits_flatten = logits.reshape(-1, logits.size(2))
            targets_flatten = y.reshape(-1)
            # print(targets_flatten.shape, logits_flatten.shape)

            loss = F.cross_entropy(logits_flatten, targets_flatten)
            # print(loss)
            loss.backward()
            optimizer.step()

            running_training_loss += loss.item()
            average_training_loss = running_training_loss / batch
            training_progress_bar.set_postfix(
                batch_loss=f"{loss.item():.4f}", avg_loss=f"{average_training_loss:.4f}")

            if batch % 50 == 0 or batch == len(training_data_loader):
                training_batches.append(batch)
                training_losses.append(average_training_loss)

            if batch % validation_interval == 0 or batch == len(training_data_loader):
                model.eval()
                running_validation_loss = 0.0
                validation_progress_bar = tqdm(
                    validation_data_loader,
                    f"[Validation] Epoch {epoch + 1} / {NUM_EPOCH}"
                )

                with torch.no_grad():
                    for _, (x, y) in enumerate(validation_progress_bar, 1):
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        logits = model(x)
                        logits_flatten = logits.reshape(-1, logits.size(2))
                        targets_flatten = y.reshape(-1)
                        loss = F.cross_entropy(logits_flatten, targets_flatten)

                        running_validation_loss += loss.item()
                        average_validation_loss = running_validation_loss / \
                            len(validation_data_loader)
                        validation_progress_bar.set_postfix(
                            batch_loss=f"{loss.item():.4f}", avg_loss=f"{average_validation_loss:.4f}")

                    validation_batches.append(batch)
                    validation_losses.append(average_validation_loss)

                model.train()

        epoch_training_loss = running_training_loss/len(training_data_loader)
        # training_loss.append(epoch_training_loss)
        # epoch_validation_loss = running_validation_loss / \
        # len(validation_data_loader)
        # validation_loss.append(epoch_validation_loss)

        # print(
        #     f'Average loss for epoch {epoch + 1}: training = {epoch_training_loss:.4f} | validation = {epoch_validation_loss:.4f}\n'
        # )

    metrics = {
        "training_batches": training_batches,
        "training_losses": training_losses,
        "validation_batches": validation_batches,
        "validation_losses": validation_losses
    }

    return model, metrics


def train_validate_and_save(label, name, tokenizer, train_loader, valid_loader):
    print("=" * 50)
    print("#" * 10, label.upper(), "#" * 10)
    print("-" * 50)

    model_path = f"./artifact/model/model_{name}.pth"
    metrics_path = f"./artifact/model/metrics_{name}.json"

    if os.path.exists(model_path) and os.path.exists(metrics_path):
        print(f"Model and metrics already exist for {name}")
        model = DecoderTransformer(
            vocabulary_size=tokenizer.get_vocab_size()
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        print(f"Training model for {label}...")
        model, metrics = train_and_validate(
            tokenizer,
            train_loader,
            valid_loader
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    print("-" * 50)
    print("=" * 50)

    return model, metrics


# %%
model_char, metrics_char = train_validate_and_save(
    "Character Tokenizer",
    "char",
    character_tokenizer,
    training_data_loader_char,
    validation_data_loader_char
)

# %%
model_bpe_small, metrics_bpe_small = train_validate_and_save(
    "BPE Tokenizer (small)",
    "bpe_small",
    bpe_tokenizer_small,
    training_data_loader_bpe_small,
    validation_data_loader_bpe_small
)

# %%
model_bpe_big, metrics_bpe_big = train_validate_and_save(
    "BPE Tokenizer (big)",
    "bpe_big",
    bpe_tokenizer_big,
    training_data_loader_bpe_big,
    validation_data_loader_bpe_big
)



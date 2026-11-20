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
from IPython.display import display
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import (BATCH_SIZE, BPE_VOCAB_SIZE_BIG, BPE_VOCAB_SIZE_SMALL,
                 CONTEXT_LENGTH, EOS_ID, LANGUAGES, LANGUAGES_LABEL_MAPPING,
                 LEARNING_RATE, LOCAL_RUN, LOCAL_SAMPLE_SIZE, NUM_EPOCHS,
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
tokenizer_directory = "./artifact/tokenizer"

os.makedirs(tokenizer_directory, exist_ok=True)


character_tokenizer_file = f"{tokenizer_directory}/character-tokenizer.json"
tokenizer_corpus_file = "./data/tokenizer/balanced.txt"

if os.path.exists(character_tokenizer_file):
    print(f"Tokenizer already exists at {character_tokenizer_file}.")
    character_tokenizer = CharacterTokenizer.from_file(
        character_tokenizer_file
    )
else:
    character_tokenizer = CharacterTokenizer(tokenizer_corpus_file)
    character_tokenizer.save(character_tokenizer_file)

print(f"Total vocab size: {len(character_tokenizer.vocab):,d}")
print(character_tokenizer.vocab[:10])

sample_encoding = character_tokenizer.encode("hello")

print(
    sample_encoding.ids,
    sample_encoding.tokens,
    character_tokenizer.decode(sample_encoding.ids)
)

print(character_tokenizer.encode("✨").ids, UNK_ID)


# %% [markdown]
# ### BPE Tokenizer

# %%
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
    BPE_VOCAB_SIZE_SMALL, tokenizer_corpus_file, f"{tokenizer_directory}/bpe-tokenizer-small.json")
bpe_tokenizer_big = train_bpe(
    BPE_VOCAB_SIZE_BIG, tokenizer_corpus_file, f"{tokenizer_directory}/bpe-tokenizer-big.json")

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
    sample_size = LOCAL_SAMPLE_SIZE if LOCAL_RUN else None

    for language in LANGUAGES:
        with open(f"./data/{type}/{language}.txt", "r", encoding="utf-8") as f:
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
        total_unk = all_token_ids.count(UNK_ID)

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
    print("=" * 10, f"{LANGUAGES_LABEL_MAPPING[language].upper()}", "=" * 10)
    print(sample)
    print(f"Character Tokenizer:\n{character_tokenizer.encode(sample).tokens}")
    print(
        f"BPE Tokenizer (small):\n{bpe_tokenizer_small.encode(sample).tokens}"
    )
    print(f"BPE Tokenizer (big):\n{bpe_tokenizer_big.encode(sample).tokens}")
    print("\n", "-" * 50, "\n")

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
    with open(f"./data/train/{language}.txt", "r", encoding="utf-8") as f:
        train_dataset[language] = [line.strip() for line in f]


def create_data_loader(dataset, tokenizer, type="train", shuffle=True, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE):
    type_label = "validation" if type == "valid" else f"{type}ing"
    is_train = type == "train"

    data = []

    for sentences in dataset.values():
        data.extend(sentences)

    print(
        f"Total {type_label} sentences across {len(dataset)} languages: {len(data):,d}")

    if is_train:
        random.seed(RANDOM_SEED)
        random.shuffle(data)

    print(f"Sample {type_label} data: {data[:10]}")

    encodings = tokenizer.encode_batch(data)
    token_ids = []

    for encoding in encodings:
        token_ids.extend(encoding.ids)
        token_ids.append(EOS_ID)

    print(f"Total {type_label} tokens: {len(token_ids):,d}")

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

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    training_batches = []
    training_losses = []
    validation_batches = []
    validation_losses = []

    validation_interval = len(training_data_loader) // 4

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_training_loss = 0.0

        training_progress_bar = tqdm(
            training_data_loader,
            f"[Training] Epoch {epoch + 1} / {NUM_EPOCHS}"
        )

        for batch, (x, y) in enumerate(training_progress_bar, 1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)

            logits_flatten = logits.reshape(-1, logits.size(2))
            targets_flatten = y.reshape(-1)

            loss = F.cross_entropy(logits_flatten, targets_flatten)
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
                    f"[Validation] Epoch {epoch + 1} / {NUM_EPOCHS}"
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


# %% [markdown]
# ### Fair Comparisons
#
# To keep things simple, we have decided to control the training by equal corpus passes. We will train all the models by running 1 full epoch.
#
# This allows the model to run through the entire dataset without any truncation.
#
# However, the tradeoff is that character tokenizers require a lot more tokens for the same sentence than the BPE tokenizers. This means that the character tokenizers will take longer to train than the BPE tokenizers. Also, with the context length of $128$, the character model corresponds to $128$ characters, which is barely enough for one full sentence. This will cause truncation of the sentences.

# %% [markdown]
# ### Model Size

# %%
def inspect(model: DecoderTransformer, name):
    embedding_dimension = model.embedding.embedding_dim
    vocabulary_size = model.vocabulary_size
    context_length = model.context_length
    input_embeddings_parameters = model.embedding.weight.numel()
    output_parameters = model.lm_head.weight.numel()
    total_parameters = sum(p.numel() for p in model.parameters())

    return {
        "Model name": name,
        "Embedding dimension": embedding_dimension,
        "Vocabulary size": vocabulary_size,
        "Input embeddings parameters": input_embeddings_parameters,
        "Output parameters": output_parameters,
        "Context length": context_length,
        "Total parameters": total_parameters
    }


model_size_df = pd.DataFrame([
    inspect(model_char, "Character"),
    inspect(model_bpe_small, "BPE (small)"),
    inspect(model_bpe_big, "BPE (big)")
])

display(model_size_df)

# %% [markdown]
# ### Training
#
# #### Learning curves

# %%

plt.figure(figsize=(11, 6))

plot_configs = [
    ("Character", metrics_char, "blue", "o"),
    ("Small BPE", metrics_bpe_small, "orange", "s"),
    ("Big BPE", metrics_bpe_big, "green", "^"),
]

for label, metrics, color, marker in plot_configs:
    plt.plot(
        metrics["training_batches"],
        metrics["training_losses"],
        label=f"{label} (Train)",
        color=color,
        alpha=0.8
    )
    plt.plot(
        metrics["validation_batches"],
        metrics["validation_losses"],
        label=f"{label} (Valid)",
        color=color,
        marker=marker,
        linestyle="--"
    )

plt.xlabel("Optimization Steps (Batches)", fontsize=11)
plt.ylabel("Cross-Entropy Loss", fontsize=11)
plt.title(
    "Training & Validation Loss Curves", fontsize=13)
plt.legend(frameon=True, fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

plt.savefig("./artifact/model/learning_curves.png", dpi=300)
plt.show()


# %% [markdown]
# #### Architecture
#
# - **Layers**: 2 decoder layers
# - **Hidden dimension**: 256
# - **Attention heads**: 4
# - **Feed-forward dimension**: 1024
# - **Context length**: 128 tokens
# - **Dropout**: 0.1
# - **Optimizer**: AdamW with a learning rate of 0.0001
# - **Batch size**: 32
# - **Training duration**: 1 epoch
#
# We also enabled weight tying between the input embedding and the output layer to keep things efficient.

# %% [markdown]
# ## Part 3: Evaluation and Analysis
#
# ### Comparing Models
#
# We have decided to use Bits Per Character (BPC) as our metric for comparing the models. BPC is a measure of the average number of bits required to represent a character in a given text. This directly solves the problem of ordinary token-level perplexity is not directly comparable between models using different tokenizers because each tokenizer breaks down the text into a different number of tokens.

# %%
def compute_bpc(model, data_loader, total_characters):
    model.eval()

    total_loss_nats = 0.0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            total_loss_nats += loss.item()

    total_loss_bits = total_loss_nats / math.log(2)

    return total_loss_bits / total_characters


# %%
test_dataset = load_dataset("test")

test_characters = {
    language: sum(len(line) for line in sentences)
    for language, sentences in test_dataset.items()
}

results = []
models = {
    "Character": (model_char, character_tokenizer),
    "BPE (small)": (model_bpe_small, bpe_tokenizer_small),
    "BPE (big)": (model_bpe_big, bpe_tokenizer_big)
}

for model_name, (model, tokenizer) in models.items():
    for language in LANGUAGES:
        loader = create_data_loader(
            {language: test_dataset[language]},
            tokenizer,
            type="test"
        )
        bpc = compute_bpc(model, loader, test_characters[language])
        results.append(
            {
                "Model": model_name,
                "Language": LANGUAGES_LABEL_MAPPING[language],
                "BPC": round(bpc, 4),
            }
        )

results_df = pd.DataFrame(results)
display(results_df)

# %%
pivot_df = results_df.pivot(index="Language", columns="Model", values="BPC")
ax = pivot_df.plot(kind="bar", figsize=(8, 4), rot=0, edgecolor="black")
ax.set_ylabel("BPC")
ax.set_title("BPC by Model and Language (Lower is Better)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("./artifact/model/bpc.png", dpi=300)
plt.show()


# %% [markdown]
# ### Who Gets the Vocabulary?

# %%
def analyze_vocabulary(tokenizer, dataset, threshold=0.9):
    vocab_size = tokenizer.get_vocab_size()

    counts = {}
    for language, sentences in dataset.items():
        counts[language] = Counter(
            token_id
            for encoding in tokenizer.encode_batch(sentences)
            for token_id in encoding.ids
        )

    stats = {
        "English": 0,
        "Turkish": 0,
        "Chinese": 0,
        "Shared": 0,
        "Unused": 0,
        "Total": vocab_size,
    }

    for token_id in range(vocab_size):
        total = sum(counts[language][token_id] for language in LANGUAGES)

        if total == 0:
            stats["Unused"] += 1
        else:
            for language in LANGUAGES:
                if counts[language][token_id] / total >= threshold:
                    stats[LANGUAGES_LABEL_MAPPING[language]] += 1
                    break
            else:
                # runs only if no language reached the threshold
                stats["Shared"] += 1

    return stats


vocabulary_results = []

for name, tokenizer in tokenizers.items():
    stats = analyze_vocabulary(tokenizer, train_dataset)
    row = {"Tokenizer": name}
    row.update(stats)
    vocabulary_results.append(row)

vocabulary_results_df = pd.DataFrame(vocabulary_results)
display(vocabulary_results_df)

# %% [markdown]
# #### BPE increment from Character

# %%
columns = ["English", "Turkish", "Chinese", "Shared"]
character_stats = vocabulary_results[0]
merges_data = []

for stat in vocabulary_results[1:]:
    diff = {
        column: stat[column] - character_stats[column]
        for column in columns
    }
    total = sum(diff.values())

    row = {"Tokenizer": stat["Tokenizer"]}
    for column, count in diff.items():
        row[column] = f"{count:,d} ({count / total:.2%})"
    row["Total"] = f"{total:,d}"

    merges_data.append(row)

merges_df = pd.DataFrame(merges_data)
display(merges_df)

# %% [markdown]
# 1. For the Character Tokenizer, it started off with $7,648$ unique characters. Chinese alone took up more than $7,000$ of them. This again reinforces what was previously mentioned that Chinese characters are tokens with meaning on their own as they do not make up with alphabet. This results in a very large vocabulary size.
# 2. Chinese does most of the merges as the slots of merges increases. In BPE (small), the merges are spread out with the shared tokens being the most. In BPE (large), the merges are more concentrated in Chinese.
# 3. Despite English and Turkish both use Latin alphabets, Turkish has more merges than English ($6,712$ vs $4,020$ in BPE - big). For both BPE (small) and BPE (large), English has the least number of merges.

# %% [markdown]
# ### Relating Tokenization to Model Behavior

# %% [markdown]
# ### Error Analysis
#

# %%
random.seed(RANDOM_SEED)

for language in LANGUAGES:
    sentence = random.choice(test_dataset[language])
    print("=" * 50)
    print(f"[{LANGUAGES_LABEL_MAPPING[language]}]")
    print(f"Sentence: {sentence}")
    print("-" * 50)

    for model_name, (model, tokenizer) in models.items():
        encoding = tokenizer.encode(sentence)

        # truncate to context length
        ids = encoding.ids[:CONTEXT_LENGTH]
        tokens = encoding.tokens[:CONTEXT_LENGTH]

        x = torch.tensor([ids[:-1]], device=DEVICE)
        y = torch.tensor([ids[1:]], device=DEVICE)

        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits.squeeze(0), y.squeeze(0)).item()

        print(f"\n{model_name} ({len(tokens)} tokens, Loss: {loss:.4f}):")
        print(f"Tokens: {tokens}")

    print("=" * 50)
    print()

# %% [markdown]
# ### Conclusion

# %% [markdown]
#

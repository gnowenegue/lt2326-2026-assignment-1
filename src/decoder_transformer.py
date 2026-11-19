import math

import torch
from torch import nn

TRANSFORMER_LAYER = 2
HIDDEN_DIMENSION = 256
ATTENTION_HEAD = 4
FEED_FORWARD_DIMENSION = 1024
DROPOUT = 0.1
CONTEXT_LENGTH = 128
BATCH_SIZE = 32


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dimension: int, maximum_sequence_length: int = 5000):
        super().__init__()
        positional_encodings = torch.zeros(
            maximum_sequence_length, hidden_dimension)

        positions = torch.arange(
            0, maximum_sequence_length, dtype=torch.float).unsqueeze(1)

        division_term = torch.exp(
            torch.arange(0, hidden_dimension, 2).float() *
            (-math.log(10000.0) / hidden_dimension)
        )

        positional_encodings[:, 0::2] = torch.sin(positions * division_term)
        positional_encodings[:, 1::2] = torch.cos(positions * division_term)

        self.register_buffer(
            'positional_encodings',
            positional_encodings.unsqueeze(0)
        )

    def forward(self, token_embeddings):
        sequence_length = token_embeddings.size(1)
        return token_embeddings + self.positional_encodings[:, :sequence_length]


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # multi-head attention sublayer
        self.attention = nn.MultiheadAttention(
            embed_dim=HIDDEN_DIMENSION,
            num_heads=ATTENTION_HEAD,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(DROPOUT)
        self.attention_normalization = nn.LayerNorm(HIDDEN_DIMENSION)

        # FFN sublayer
        self.feed_forward = nn.Sequential(
            nn.Linear(HIDDEN_DIMENSION, FEED_FORWARD_DIMENSION),
            nn.ReLU(),
            nn.Linear(FEED_FORWARD_DIMENSION, HIDDEN_DIMENSION),
        )
        self.feed_forward_dropout = nn.Dropout(DROPOUT)
        self.feed_forward_normalization = nn.LayerNorm(HIDDEN_DIMENSION)

    def forward(self, x):
        sequence_length = x.size(1)

        causal_attention_mask = nn.Transformer.generate_square_subsequent_mask(
            sequence_length,
            device=x.device
        )

        # multi-head attention sublayer
        attention_output, _ = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_attention_mask,
            need_weights=False
        )
        attention_output = self.attention_dropout(attention_output)
        x = x + attention_output
        x = self.attention_normalization(x)

        # FFN sublayer
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.feed_forward_dropout(feed_forward_output)
        x = x + feed_forward_output
        x = self.feed_forward_normalization(x)

        return x


class DecoderTransformer(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.context_length = CONTEXT_LENGTH

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=HIDDEN_DIMENSION
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            hidden_dimension=HIDDEN_DIMENSION,
            maximum_sequence_length=CONTEXT_LENGTH
        )
        self.dropout = nn.Dropout(DROPOUT)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer() for _ in range(TRANSFORMER_LAYER)
        ])

        self.lm_head = nn.Linear(HIDDEN_DIMENSION, vocabulary_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = x * math.sqrt(HIDDEN_DIMENSION)

        x = self.positional_encoding(x)

        x = self.dropout(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)

        logits = self.lm_head(x)

        return logits

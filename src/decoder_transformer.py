import math

import torch
from torch import nn

from .constants import (ATTENTION_HEAD, CONTEXT_LENGTH, DROPOUT,
                        FEED_FORWARD_DIMENSION, HIDDEN_DIMENSION,
                        TRANSFORMER_LAYER)


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
            "positional_encodings",
            positional_encodings.unsqueeze(0)
        )

    def forward(self, token_embeddings):
        sequence_length = token_embeddings.size(1)
        return token_embeddings + self.positional_encodings[:, :sequence_length]


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=HIDDEN_DIMENSION,
        num_heads=ATTENTION_HEAD,
        ff_dimension=FEED_FORWARD_DIMENSION,
        dropout=DROPOUT
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_normalization = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dimension),
            nn.ReLU(),
            nn.Linear(ff_dimension, embed_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_normalization = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        attention_output, _ = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=mask,
            need_weights=False
        )
        attention_output = self.attention_dropout(attention_output)
        x = x + attention_output
        x = self.attention_normalization(x)

        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.feed_forward_dropout(feed_forward_output)
        x = x + feed_forward_output
        x = self.feed_forward_normalization(x)

        return x


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dim=HIDDEN_DIMENSION,
        num_layers=TRANSFORMER_LAYER,
        context_length=CONTEXT_LENGTH,
        dropout=DROPOUT
    ):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            hidden_dimension=embedding_dim,
            maximum_sequence_length=context_length
        )
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embedding_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(embedding_dim, vocabulary_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = x * math.sqrt(self.embedding_dim)

        x = self.positional_encoding(x)

        x = self.dropout(x)

        sequence_length = x.size(1)

        causal_attention_mask = nn.Transformer.generate_square_subsequent_mask(
            sequence_length,
            device=x.device
        )

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask=causal_attention_mask)

        logits = self.lm_head(x)

        return logits

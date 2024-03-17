import math

import torch
from torch import Tensor
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, model_dim: int, vocabular_size: int) -> None:
        super().__init__()

        self.embeddings = nn.Embedding(vocabular_size, model_dim)
        self.model_dim = model_dim

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embeddings(tokens) * math.sqrt(self.model_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, seq_len: int, dropout: float) -> None:
        super().__init__()

        self.embeddings = nn.Embedding(seq_len, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, weights: Tensor) -> Tensor:
        weights = (
            weights +
            self.embeddings.weight[:weights[1], :]
        )

        return self.dropout(weights)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, inner_dim: int, dropout: float) -> None:
        super().__init__()

        self.input_linear = nn.Linear(model_dim, inner_dim, bias=False)
        self.output_linear = nn.Linear(inner_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, weights: Tensor) -> Tensor:
        return self.output_linear(
            self.dropout(torch.relu(self.input_linear(weights)))
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()

        assert model_dim % n_heads == 0

        self.n_heads = n_heads
        self.head_size = model_dim // n_heads
        self.query_weights = nn.Linear(model_dim, model_dim)
        self.key_weights = nn.Linear(model_dim, model_dim)
        self.values_weights = nn.Linear(model_dim, model_dim)
        self.out_weights = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout)

    def _to_heads_view(self, data: Tensor) -> Tensor:
        return data.view(
            data.shape[0], data.shape[1], self.n_heads, self.head_size,
        ).transpose(1, 2)

    def calc_attention(
        self, query: Tensor,
        key: Tensor,
        values: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_size)
        if mask is not None:
            attention = attention.masked_fill_(mask == 0, -1e9)
        attention = self.dropout(attention)

        return attention @ values

    def forward(
        self, query_in: Tensor,
        key_in: Tensor,
        values_in: Tensor,
        mask: Tensor | None = None,
    ) -> None:
        query = self.query_weights(query_in)
        key = self.key_weights(key_in)
        values = self.values_weights(values_in)

        query_heads = self._to_heads_view(query)
        key_heads = self._to_heads_view(key)
        values_heads = self._to_heads_view(values)

        attention = self.calc_attention(
            query_heads, key_heads, values_heads, mask
        )

        concat_attention = attention.transpose(1, 2).contiguous().view(
            attention.shape[0], -1, self.head_size * self.n_heads,
        )

        return self.out_weights(concat_attention)


class LayerNorm(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps = eps
        self.alpha = torch.ones(n_features)
        self.bias = torch.zeros(n_features)

    def forward(self, weights: Tensor) -> Tensor:
        mean = weights.mean(dim=-1, keepdim=True)
        std = weights.std(dim=-1, keepdim=True)

        return self.alpha * (weights - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):
    def __init__(self, n_features: int, dropout: float) -> None:
        super().__init__()

        self.norm = LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, weights: Tensor, sublayer: nn.Module) -> None:
        return weights + self.dropout(sublayer(self.norm(weights)))


class EncoderBlock(nn.Module):
    def __init__(
        self, n_features: int,
        mha_layer: MultiHeadAttention,
        ff_layer: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()

        self.mha_layer = mha_layer
        self.ff_layer = ff_layer
        self.mha_residual = ResidualConnection(n_features, dropout)
        self.ff_residual = ResidualConnection(n_features, dropout)

    def forward(self, weights: Tensor, mask: Tensor | None = None) -> Tensor:
        weights = self.mha_residual(weights, lambda w: self.mha_layer(
            w, w, w, mask,
        ))

        weights = self.ff_residual(weights, self.ff_layer)

        return weights


class DecoderBlock(nn.Module):
    def __init__(
        self, n_features: int,
        mha_layer: MultiHeadAttention,
        ca_layer: MultiHeadAttention,
        ff_layer: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()

        self.mha_layer = mha_layer
        self.ca_layer = ca_layer
        self.ff_layer = ff_layer
        self.mha_residual = ResidualConnection(n_features, dropout)
        self.ca_residual = ResidualConnection(n_features, dropout)
        self.ff_residual = ResidualConnection(n_features, dropout)

    def forward(
        self, weights: Tensor,
        encoded_weights: Tensor,
        mask: Tensor,
        encoder_mask: Tensor,
    ) -> Tensor:
        weights = self.mha_residual(weights, lambda w: self.mha_layer(
            w, w, w, mask
        ))
        weights = self.ca_residual(weights, lambda w: self.ca_layer(
            w, encoded_weights, encoded_weights, encoder_mask,
        ))
        weights = self.ff_residual(weights, self.ff_layer)

        return weights


class MultiBlockPipeline(nn.Module):
    def __init__(self, n_features: int, blocks: nn.ModuleList) -> None:
        super().__init__()

        self.blocks = blocks
        self.norm = LayerNorm(n_features)

    def forward(self, weights: Tensor, *args, **kwargs) -> Tensor:
        for block in self.blocks:
            weights = block(weights, *args, **kwargs)

        return self.norm(weights)


class FinalLinear(nn.Module):
    def __init__(self, model_dim: int, vocabular_size: int) -> None:
        super().__init__()

        self.final_linear = nn.Linear(model_dim, vocabular_size)

    def forward(self, weights: Tensor) -> Tensor:
        return self.final_linear(weights)


class FinalSoftmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, weights: Tensor) -> Tensor:
        return torch.softmax(weights)


class Transformer(nn.Module):
    def __init__(
        self, input_embeddings: EmbeddingLayer,
        output_embeddings: EmbeddingLayer,
        input_pos_embed: PositionalEncoding,
        output_pos_embed: PositionalEncoding,
        encoder: MultiBlockPipeline,
        decoder: MultiBlockPipeline,
        final_linear: FinalLinear,
        final_softmax: FinalSoftmax,
    ) -> None:
        self.input_embeddings = input_embeddings
        self.output_embeddings = output_embeddings
        self.input_pos_embed = input_pos_embed
        self.output_pos_embed = output_pos_embed
        self.encoder = encoder
        self.decoder = decoder
        self.final_linear = final_linear
        self.final_softmax = final_softmax

    def encode(self, tokens: Tensor, mask: Tensor) -> Tensor:
        embeddings = self.input_embeddings(tokens)
        weights = self.input_pos_embed(embeddings)

        return self.encoder(weights, mask)

    def decode(
        self, output_tokens: Tensor,
        encode_weights: Tensor,
        output_mask: Tensor,
        encode_mask: Tensor,
    ) -> Tensor:
        embeddings = self.output_embeddings(output_tokens)
        weights = self.output_pos_embed(embeddings)

        return self.decoder(weights, encode_weights, output_mask, encode_mask)

    def final_steps(self, weights: Tensor) -> Tensor:
        weights = self.final_linear(weights)

        return self.final_softmax(weights)


def build_transformer(
    source_vocabular_size: int,
    targer_vocabular_size: int,
    source_seq_len: int,
    target_seq_len: int,
    model_dim: int = 512,
    inner_dim: int = 2048,
    dropout: float = 0.1,
    n_blocks: int = 6,
    n_heads: int = 8,
) -> Transformer:
    input_embeddings = EmbeddingLayer(model_dim, source_vocabular_size)
    output_embeddings = EmbeddingLayer(model_dim, targer_vocabular_size)

    input_pos_embed = PositionalEncoding(model_dim, source_seq_len, dropout)
    output_pos_embed = PositionalEncoding(model_dim, target_seq_len, dropout)

    encoder_blocks = []
    for _ in range(n_blocks):
        encoder_mha_layer = MultiHeadAttention(model_dim, n_heads, dropout)
        encoder_ff_layer = FeedForward(model_dim, inner_dim, dropout)

        encoder_blocks.append(
            EncoderBlock(model_dim, encoder_mha_layer, encoder_ff_layer)
        )

    decoder_blocks = []
    for _ in range(n_blocks):
        decoder_mha_layer = MultiHeadAttention(model_dim, n_heads, dropout)
        decoder_ca_layer = MultiHeadAttention(model_dim, n_heads, dropout)
        decoder_ff_layer = FeedForward(model_dim, inner_dim, dropout)

        decoder_blocks.append(
            DecoderBlock(
                n_features=model_dim,
                mha_layer=decoder_mha_layer,
                ca_layer=decoder_ca_layer,
                ff_layer=decoder_ff_layer,
                dropout=dropout
            )
        )

    encoder = MultiBlockPipeline(model_dim, encoder_blocks)
    decoder = MultiBlockPipeline(model_dim, decoder_blocks)

    final_linear = FinalLinear(model_dim, targer_vocabular_size)
    final_softmax = FinalSoftmax()

    return Transformer(
        input_embeddings=input_embeddings,
        output_embeddings=output_embeddings,
        input_pos_embed=input_pos_embed,
        output_pos_embed=output_pos_embed,
        encoder=encoder,
        decoder=decoder,
        final_linear=final_linear,
        final_softmax=final_softmax,
    )

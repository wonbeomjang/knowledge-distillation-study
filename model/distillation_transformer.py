from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def make_attention(x: torch.Tensor, size=None):
    x = F.normalize(x.pow(2).mean(1))
    if size:
        x = F.adaptive_avg_pool2d(x, size)
    return x


def make_patch(x: torch.Tensor, embedding_size: int):
    x = make_attention(x, int(embedding_size ** (1/2))).view(x.size(0), -1)
    assert x.size(1) == embedding_size
    return x


class MakePatchBlock(nn.Module):
    def __init__(self, embedding_size):
        super(MakePatchBlock, self).__init__()
        self.embedding_size = embedding_size

    def forward(self, x):
        attention = torch.stack([make_patch(t, self.embedding_size) for t in x], dim=1)
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int = 1, dropout_p: float = .0):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_p)

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

    def forward(self, queries, keys, values):
        queries = rearrange(self.queries(queries), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(keys), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(values), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        scaling = self.embed_size ** (1 / 2)
        attention = F.softmax(energy, dim=-1) / scaling
        attention = self.dropout(attention)

        out = torch.einsum("bhal, bhlv -> bhav", attention, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, embed_size: int, expansion: int = 4, drop_p: float = .0):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(embed_size, embed_size * expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(embed_size * expansion, embed_size)
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_size: int, drop_p: int = .0, forward_expansion: int = 4, forward_drop_p: float = .0,
                 **kwargs):
        super(TransformerEncoderBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(drop_p)
        self.multi_head_attention = MultiHeadAttention(embed_size, **kwargs)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embed_size),
            FeedForwardBlock(embed_size, expansion=forward_expansion, drop_p=forward_drop_p),
            nn.Dropout(drop_p)
        )

    def forward(self, x):
        x1 = self.layer_norm(x)
        x1 = self.multi_head_attention(x1, x1, x1)
        x += self.dropout(x1)
        x += self.feed_forward(x)
        return x


def get_positional_encoding(embed_size, max_len, device):
    positional_encoding = torch.zeros((max_len, embed_size), requires_grad=False, device=device)
    pos = torch.arange(0, max_len, requires_grad=False, device=device).float().unsqueeze(dim=1)
    _2i = torch.arange(0, embed_size, step=2, requires_grad=False, device=device).float()
    positional_encoding[:, ::2] = torch.sin(pos / (10000 * (_2i / embed_size)))
    positional_encoding[:, 1::2] = torch.sin(pos / (10000 * (_2i / embed_size)))

    return positional_encoding


class AttentionEncoder(nn.Module):
    def __init__(self, embedding_size: int, device: torch.device, depth: int = 4, max_len: int = 4, **kwargs):
        """
        :param embedding_size: input embedding size. it must be larger than attention map size
        :param depth: number of encoder block
        :param max_len: max number of attention feature
        """
        super(AttentionEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.encoders = nn.Sequential(*[
            TransformerEncoderBlock(embed_size=self.embedding_size, **kwargs) for i in range(depth)
        ])
        self.positional_encoding = get_positional_encoding(self.embedding_size, max_len, device)

    def forward(self, attention):
        attention += self.positional_encoding[:attention.size(1), :]
        attention = self.encoders(attention)

        return attention


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_size: int, drop_p: int = .0, forward_expansion: int = 4, forward_drop_p: float = .0,
                 **kwargs):
        super(TransformerDecoderBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(drop_p)
        self.dropout_self_attention = nn.Dropout(drop_p)
        self.multi_head_self_attention = MultiHeadAttention(embed_size, **kwargs)
        self.multi_head_attention = MultiHeadAttention(embed_size, **kwargs)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embed_size),
            FeedForwardBlock(embed_size, expansion=forward_expansion, drop_p=forward_drop_p),
            nn.Dropout(drop_p)
        )

    def forward(self, x: Union[torch.Tensor, torch.Tensor]):
        x, enc_src = x
        x1 = self.layer_norm(x)
        x1 = self.multi_head_self_attention(x1, x1, x1)
        x += self.dropout_self_attention(x1)

        x1 = self.layer_norm(x)
        x1 = self.multi_head_attention(x1, enc_src, enc_src)
        x += self.dropout(x1)

        x += self.feed_forward(x)
        return x, enc_src


class AttentionDecoder(nn.Module):
    def __init__(self, embedding_size: int, device: torch.device, depth: int = 4, max_len: int = 4, **kwargs):
        """
        :param embedding_size: input patch size. it must be larger than attention map size
        :param depth: number of encoder block
        :param max_len: max number of attention feature
        """
        super(AttentionDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.decoders = nn.Sequential(*[
            TransformerDecoderBlock(embed_size=self.embedding_size, **kwargs) for i in range(depth)
        ])
        self.positional_encoding = get_positional_encoding(self.embedding_size, max_len, device)

        self.linear = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, attention, enc_src: torch.Tensor):
        attention += self.positional_encoding[:attention.size(1), :]
        attention, enc_src = self.decoders((attention, enc_src))
        attention = self.linear(attention)
        return attention


class ReshapeBlock(nn.Module):
    def __init__(self, embedding_size):
        super(ReshapeBlock, self).__init__()
        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor):
        print(x.size())
        return x.view((x.size(0), int(self.embedding_size ** (1/2)), int(self.embedding_size ** (1/2))))


class DistillationTransformer(nn.Module):
    def __init__(self, embedding_size: int, device: torch.device, **kwargs):
        """
        :param embedding_size: input patch size. it must be larger than attention map size
        :param device: torch.device to run
        """
        super(DistillationTransformer, self).__init__()
        self.attention_block = MakePatchBlock(embedding_size)
        self.encoder = AttentionEncoder(embedding_size, device, **kwargs)
        self.decoder = AttentionDecoder(embedding_size, device, **kwargs)
        self.reshape_block = ReshapeBlock(embedding_size)
        self._initialize_weights()

    def forward(self, src, trg):
        src = self.attention_block(src)
        enc_src = self.encoder(src)
        x = self.decoder(self.attention_block(trg), enc_src)
        x = self.reshape_block(x)

        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



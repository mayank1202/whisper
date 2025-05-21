import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from decoding import decode as decode_function
from decoding import detect_language as detect_language_function
from transcribe import transcribe as transcribe_function

from hw_perf import *

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        self.perf_kpis = PerfKPIs()
        batchsize = x.shape[0] 
        weight_transposed = self.weight.transpose(0,1)
        for count in range(batchsize):
            mm_perf_kpis = matmul_2d_perf_kpis(x[count], weight_transposed)
            self.perf_kpis.add(mm_perf_kpis)

        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        #npu_conv1d(x,self,weight,bias)
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.num_params = 0
        self.n_head = n_head
        
        self.query = Linear(n_state, n_state)
        self.num_params += n_state * n_state

        self.key = Linear(n_state, n_state, bias=False)
        self.num_params += n_state * n_state

        self.value = Linear(n_state, n_state)
        self.num_params += n_state * n_state

        self.out = Linear(n_state, n_state)
        self.num_params += n_state * n_state

        self.perf_kpis = PerfKPIs()

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)
        self.perf_kpis.add(self.query.perf_kpis)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            self.perf_kpis.add(self.key.perf_kpis)

            v = self.value(x if xa is None else xa)
            self.perf_kpis.add(self.value.perf_kpis)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk, sdpa_perf_kpis = self.qkv_attention(q, k, v, mask)
        self.perf_kpis.add(sdpa_perf_kpis)

        out_ = self.out(wv)
        self.perf_kpis.add(self.out.perf_kpis)

        return out_, qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        
        sdpa_perf_kpis = PerfKPIs()

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        q_single_head = q[0,0,:,:]
        k_single_head = k[0,0,:,:]
        v_single_head = v[0,0,:,:]

        k_single_head_transposed = k_single_head.transpose(0,1)

        qk_perf_kpis = matmul_2d_perf_kpis(q_single_head, k_single_head_transposed)
        qk_single_head = torch.zeros((q_single_head.shape[0], k_single_head_transposed.shape[1]))
        sftmx_perf_kpis = softmax_perf_kpis(qk_single_head)
        av_perf_kpis = matmul_2d_perf_kpis(qk_single_head, v_single_head)

        for count in range(n_batch * self.n_head):
            sdpa_perf_kpis.add(qk_perf_kpis)
            sdpa_perf_kpis.add(sftmx_perf_kpis)
            sdpa_perf_kpis.add(av_perf_kpis)

        return out, qk, sdpa_perf_kpis


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.num_params = 0

        self.attn = MultiHeadAttention(n_state, n_head)
        self.num_params += self.attn.num_params

        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        if(cross_attention):
            self.num_params += self.cross_attn.num_params

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )

        self.num_params += (n_state * n_mlp) + (n_mlp * n_state)
        self.mlp_ln = LayerNorm(n_state)

        self.perf_kpis = PerfKPIs()

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        self.perf_kpis.reset()

        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        mha_perf_kpis = self.attn.perf_kpis
        self.perf_kpis.add(mha_perf_kpis) #Ignoring ADD operation since it will be small

        if self.cross_attn:
            self.cross_attn.perf_kpis.reset()
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
            crossattn_perf_kpis = self.cross_attn.perf_kpis
            self.perf_kpis.add(crossattn_perf_kpis) #Ignoring ADD operation since it will be small

        x = x + self.mlp(self.mlp_ln(x))
        MLP_perf_kpis = mlp_perf_kpis(x, self.mlp)
        self.perf_kpis.add(MLP_perf_kpis) #Ignoring ADD operation since it will be small

        lnorm_perf_kpis = layernorm_perf_kpis(x)
        self.perf_kpis.add(lnorm_perf_kpis)

        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.num_params = 0

        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.num_params += n_mels * n_state * 3

        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.num_params += n_state * n_state * 3

        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.num_params += n_ctx * n_state

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.num_params += n_layer * self.blocks[0].num_params

        self.ln_post = LayerNorm(n_state)

        self.perf_kpis = PerfKPIs()

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        self.perf_kpis.reset()

        x = F.gelu(self.conv1(x))
        conv1_perf_kpis = conv1d_perf_kpis(x, self.conv1)
        self.perf_kpis.add(conv1_perf_kpis)

        x = F.gelu(self.conv2(x))
        conv2_perf_kpis = conv1d_perf_kpis(x, self.conv2)
        self.perf_kpis.add(conv2_perf_kpis)

        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            block.perf_kpis.reset()
            x = block(x)
            self.perf_kpis.add(block.perf_kpis)

        x = self.ln_post(x)
        ln_perf_kpis = layernorm_perf_kpis(x)
        self.perf_kpis.add(ln_perf_kpis)

        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.num_params = 0

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.num_params += n_vocab * n_state

        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.num_params += n_ctx * n_state

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.num_params += n_layer * self.blocks[0].num_params

        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

        self.perf_kpis = PerfKPIs()

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        y = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        token_embedding_perf_kpis = embedding_perf_kpis(x, self.token_embedding)
        self.perf_kpis.add(token_embedding_perf_kpis)
        #Ignore positional_embedding
        
        x = y.to(xa.dtype)

        for block in self.blocks:
            block.perf_kpis.reset()
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
            self.perf_kpis.add(block.perf_kpis)

        x = self.ln(x)
        ln_perf_kpis = layernorm_perf_kpis(x)
        self.perf_kpis.add(ln_perf_kpis)

        weight_transposed = torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        logits = (x @ weight_transposed).float()

        batchsize = x.shape[0]
        for count in range(batchsize):
            output_projection_perf_kpis = matmul_2d_perf_kpis(x[count], weight_transposed)
            self.perf_kpis.add(output_projection_perf_kpis)

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)
        self.perf_kpis = PerfKPIs()
        self.num_params = self.encoder.num_params + self.decoder.num_params

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

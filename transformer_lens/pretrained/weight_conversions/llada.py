from typing import cast

import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_llada_weights(hf_model, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = hf_model.model.transformer.wte.weight

    # Some models with the Llama architecture use Grouped Query Attention, and so for these we need to modify
    # the state dict keys for the K/V attention weight/biases, prepending "_" to the key names.
    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""
    # need a cast since MyPy isn't smart enough to realize that using_gqa implies n_key_value_heads is not None
    n_kv_heads = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads)

    # llama has no biases anywhere and deals with everything else roughly like
    # GPTNeoX with different names

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = hf_model.model.transformer.blocks[l].attn_norm.weight

        W_Q = hf_model.model.transformer.blocks[l].q_proj.weight
        W_K = hf_model.model.transformer.blocks[l].k_proj.weight
        W_V = hf_model.model.transformer.blocks[l].v_proj.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
            W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
            W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            n_kv_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            n_kv_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )

        W_O = hf_model.model.transformer.blocks[l].attn_out.weight

        if not cfg.load_in_4bit:
            W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        state_dict[f"blocks.{l}.ln2.w"] = hf_model.model.transformer.blocks[l].ff_norm.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            state_dict[f"blocks.{l}.mlp.W_in"] = hf_model.model.transformer.blocks[l].up_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_gate"] = hf_model.model.transformer.blocks[l].ff_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_out"] = hf_model.model.transformer.blocks[l].ff_out.weight.T
        else:
            state_dict[f"blocks.{l}.mlp.W_in"] = hf_model.model.transformer.blocks[l].up_proj.weight
            state_dict[f"blocks.{l}.mlp.W_gate"] = hf_model.model.transformer.blocks[l].ff_proj.weight
            state_dict[f"blocks.{l}.mlp.W_out"] = hf_model.model.transformer.blocks[l].ff_out.weight

        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

    state_dict["ln_final.w"] = hf_model.model.transformer.ln_f.weight

    state_dict["unembed.W_U"] = hf_model.model.transformer.ff_out.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=cfg.device)

    return state_dict
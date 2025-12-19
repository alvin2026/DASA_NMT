import torch
import torch.nn.functional as F
from fairseq.modules.multihead_attention import MultiheadAttention


class MultiheadAttentionDASA(MultiheadAttention):
    """
    fairseq 0.12.2 compatible.
    dep_bias: FloatTensor [B, T, S] aligned to src_tokens (subword level).
    modes:
      - mul: logits = logits * dep_bias
      - log: logits = logits + lam * log(dep_bias)
    """

    def __init__(self, *args, dasa_mode="mul", dasa_lambda=1.0, eps=1e-9, **kwargs):
        super().__init__(*args, **kwargs)
        self.dasa_mode = dasa_mode
        self.dasa_lambda = float(dasa_lambda)
        self.dasa_eps = float(eps)

    def forward(
        self,
        query,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
        dep_bias: torch.Tensor = None,
    ):
        # === Copied/trimmed from fairseq 0.12.2 MultiheadAttention.forward ===
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert saved_state is not None and "prev_key" in saved_state
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(value)
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )  # [B*H, T, Dh]

        if k is not None:
            src_len = k.size(0)
            k = (
                k.contiguous()
                .view(src_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )  # [B*H, S, Dh]
        else:
            src_len = saved_state["prev_key"].size(2)

        if v is not None:
            v = (
                v.contiguous()
                .view(src_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )  # [B*H, S, Dh]

        if saved_state is not None:
            # incremental decoding (not used in encoder self-attn training, but keep for completeness)
            if "prev_key" in saved_state:
                prev_k = saved_state["prev_key"]
                if static_kv:
                    k = prev_k
                else:
                    k = torch.cat([prev_k, k], dim=1)
            if "prev_value" in saved_state:
                prev_v = saved_state["prev_value"]
                if static_kv:
                    v = prev_v
                else:
                    v = torch.cat([prev_v, v], dim=1)
            saved_state["prev_key"] = k
            saved_state["prev_value"] = v
            self._set_input_buffer(incremental_state, saved_state)

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # [B*H, T, S]
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        # === Inject DASA bias right before masks + softmax ===
        if dep_bias is not None:
            # dep_bias: [B, T, S] -> [B*H, T, S]
            bias = dep_bias[:, :tgt_len, :src_len].to(attn_weights.device).to(attn_weights.dtype)
            bias = bias.unsqueeze(1).expand(bsz, self.num_heads, tgt_len, src_len).contiguous()
            bias = bias.view(bsz * self.num_heads, tgt_len, src_len)

            if self.dasa_mode == "mul":
                attn_weights = attn_weights * bias
            elif self.dasa_mode == "log":
                attn_weights = attn_weights + self.dasa_lambda * torch.log(bias.clamp_min(self.dasa_eps))
            else:
                raise ValueError(f"Unknown dasa_mode={self.dasa_mode}")

        if attn_mask is not None:
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: [B, S] True for pad
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_probs, v)  # [B*H, T, Dh]
        attn = (
            attn.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, embed_dim)
        )
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_probs_ = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            if need_head_weights:
                attn_weights_out = attn_probs_
            else:
                attn_weights_out = attn_probs_.mean(dim=1)
        else:
            attn_weights_out = None

        return attn, attn_weights_out
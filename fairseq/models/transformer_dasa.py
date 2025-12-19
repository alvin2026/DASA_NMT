from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    TransformerConfig,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from dasa_fairseq.modules.multihead_attention_dasa import MultiheadAttentionDASA


class TransformerEncoderLayerDASA(TransformerEncoderLayer):
    def __init__(self, cfg: TransformerConfig):
        super().__init__(cfg)
        self.self_attn = MultiheadAttentionDASA(
            embed_dim=cfg.encoder.embed_dim,
            num_heads=cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            dasa_mode=getattr(cfg, "dasa_mode", "mul"),
            dasa_lambda=getattr(cfg, "dasa_lambda", 1.0),
        )

    def forward(self, x, encoder_padding_mask, attn_mask=None, dep_bias=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
            dep_bias=dep_bias,
        )
        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        return x


class TransformerEncoderDASA(TransformerEncoder):
    # 重写 forward：逐层传 dep_bias
    def forward(self, src_tokens, src_lengths, dep_bias=None, **kwargs):
        x, encoder_embedding = self.forward_embedding(src_tokens)
        x = x.transpose(0, 1)  # [T,B,C]

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        attn_mask = None

        for layer in self.layers:
            x = layer(x, encoder_padding_mask, attn_mask=attn_mask, dep_bias=dep_bias)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # [T,B,C]
            "encoder_padding_mask": [encoder_padding_mask],  # [B,S]
            "encoder_embedding": [encoder_embedding],  # [B,S,C]
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }


@register_model("transformer_dasa")
class TransformerModelDASA(TransformerModel):
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        base_encoder = TransformerEncoderDASA(cfg, src_dict, embed_tokens)
        # 用 DASA layer 替换
        base_encoder.layers = base_encoder.layers.__class__([
            TransformerEncoderLayerDASA(cfg) for _ in range(cfg.encoder.layers)
        ])
        return base_encoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, dep_bias=None, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, dep_bias=dep_bias, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out


@register_model_architecture("transformer_dasa", "transformer_dasa_base")
def transformer_dasa_base(args):
    # 使用 fairseq transformer_base 默认即可；DASA 参数通过 CLI 传入
    pass
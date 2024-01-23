# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel
from unicore.modules import LayerNorm, init_bert_params
from model.transformer_encoder_with_pair import TransformerEncoderWithPair



class SimpleUniMolModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--unimol-encoder-layers", type=int, metavar="L", help="num encoder layers", default=15
        )
        parser.add_argument(
            "--unimol-encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension", default=512,
        )
        parser.add_argument(
            "--unimol-encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN", default=2048,
        )
        parser.add_argument(
            "--unimol-encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads", default=64,
        )
        parser.add_argument(
            "--unimol-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use", default="gelu",
        )
        parser.add_argument(
            "--unimol-emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings", default=0.1,
        )
        parser.add_argument(
            "--unimol-dropout", type=float, metavar="D", help="dropout probability", default=0.1
        )
        parser.add_argument(
            "--unimol-attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights", default=0.1,
        )
        parser.add_argument(
            "--unimol-activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN", default=0.0,
        )
        parser.add_argument(
            "--unimol-max-seq-len", type=int, help="number of positional embeddings to learn", default=512
        )
        parser.add_argument(
            "--unimol-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio", default=-1.0,
        )
        parser.add_argument(
            "--unimol-max-atoms", type=int, default=256
        )
        return parser

    def __init__(self, args, dictionary):
        super().__init__()
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.unimol_encoder_embed_dim, self.padding_idx
        )
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.unimol_encoder_layers,
            embed_dim=args.unimol_encoder_embed_dim,
            ffn_embed_dim=args.unimol_encoder_ffn_embed_dim,
            attention_heads=args.unimol_encoder_attention_heads,
            emb_dropout=args.unimol_emb_dropout,
            dropout=args.unimol_dropout,
            attention_dropout=args.unimol_attention_dropout,
            activation_dropout=args.unimol_activation_dropout,
            max_seq_len=args.unimol_max_seq_len,
            activation_fn=args.unimol_activation_fn,
            no_final_head_layer_norm=args.unimol_delta_pair_repr_norm_loss < 0,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.unimol_encoder_attention_heads, args.unimol_activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.num_features = args.unimol_encoder_embed_dim
        self.apply(init_bert_params)


    def forward(
        self,
        src_tokens,
        src_distance,
        src_edge_type,
    ):
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        return encoder_rep, padding_mask




class UniMolModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--unimol-encoder-layers", type=int, metavar="L", help="num encoder layers", default=15
        )
        parser.add_argument(
            "--unimol-encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension", default=512,
        )
        parser.add_argument(
            "--unimol-encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN", default=2048,
        )
        parser.add_argument(
            "--unimol-encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads", default=64,
        )
        parser.add_argument(
            "--unimol-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use", default="gelu",
        )
        parser.add_argument(
            "--unimol-pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer", default="tanh",
        )
        parser.add_argument(
            "--unimol-emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings", default=0.1,
        )
        parser.add_argument(
            "--unimol-dropout", type=float, metavar="D", help="dropout probability", default=0.1
        )
        parser.add_argument(
            "--unimol-attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights", default=0.1,
        )
        parser.add_argument(
            "--unimol-activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN", default=0.0,
        )
        parser.add_argument(
            "--unimol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers", default=0.0,
        )
        parser.add_argument(
            "--unimol-max-seq-len", type=int, help="number of positional embeddings to learn", default=512
        )
        parser.add_argument(
            "--unimol-post-ln", type=bool, help="use post layernorm or pre layernorm", default=False
        )
        parser.add_argument(
            "--unimol-masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio", default=-1.0,
        )
        parser.add_argument(
            "--unimol-masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio", default=-1.0,
        )
        parser.add_argument(
            "--unimol-masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio", default=-1.0,
        )
        parser.add_argument(
            "--unimol-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio", default=-1.0,
        )
        parser.add_argument(
            "--unimol-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio", default=-1.0,
        )
        parser.add_argument(
            "--unimol-masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio", default=-1.0,
        )
        parser.add_argument(
            "--unimol-mode",
            type=str,
            default="infer",
            choices=["train", "infer"], 
        )
        return parser

    def __init__(self, args, dictionary):
        super().__init__()
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.unimol_encoder_embed_dim, self.padding_idx
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.unimol_encoder_layers,
            embed_dim=args.unimol_encoder_embed_dim,
            ffn_embed_dim=args.unimol_encoder_ffn_embed_dim,
            attention_heads=args.unimol_encoder_attention_heads,
            emb_dropout=args.unimol_emb_dropout,
            dropout=args.unimol_dropout,
            attention_dropout=args.unimol_attention_dropout,
            activation_dropout=args.unimol_activation_dropout,
            max_seq_len=args.unimol_max_seq_len,
            activation_fn=args.unimol_activation_fn,
            no_final_head_layer_norm=args.unimol_delta_pair_repr_norm_loss < 0,
        )
        if args.unimol_masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.unimol_encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.unimol_activation_fn,
                weight=None,
            )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.unimol_encoder_attention_heads, args.unimol_activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        if args.unimol_masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.unimol_encoder_attention_heads, 1, args.unimol_activation_fn
            )
        if args.unimol_masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.unimol_encoder_attention_heads, args.unimol_activation_fn
            )
        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):

        if classification_head_name is not None:
            features_only = True

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if not features_only:
            if self.args.unimol_masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.args.unimol_masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(
                        -1, 1, 1, 1
                    )
                else:
                    atom_num = src_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                encoder_coord = coords_emb + coord_update
            if self.args.unimol_masked_dist_loss > 0:
                encoder_distance = self.dist_head(encoder_pair_rep)

        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](encoder_rep)
        if self.args.unimol_mode == 'infer':
            return encoder_rep, encoder_pair_rep
        else:
            return (
                logits,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm,
            )         

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.unimol_encoder_embed_dim,
            inner_dim=inner_dim or self.args.unimol_encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.unimol_pooler_activation_fn,
            pooler_dropout=self.args.unimol_pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


# @register_model_architecture("unimol", "unimol")
# def base_architecture(args):
#     args.encoder_layers = getattr(args, "encoder_layers", 15)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
#     args.dropout = getattr(args, "dropout", 0.1)
#     args.emb_dropout = getattr(args, "emb_dropout", 0.1)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.1)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.0)
#     args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
#     args.max_seq_len = getattr(args, "max_seq_len", 512)
#     args.activation_fn = getattr(args, "activation_fn", "gelu")
#     args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
#     args.post_ln = getattr(args, "post_ln", False)
#     args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
#     args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
#     args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
#     args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
#     args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
#     print(args.encoder_embed_dim)


# # @register_model_architecture("unimol", "unimol_base")
# def unimol_base_architecture(args):
#     base_architecture(args)

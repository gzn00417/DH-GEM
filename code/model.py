import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import dgl
import dgl.nn.pytorch as dgl_nn


class BaseSeqModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.criterion = nn.NLLLoss()

    def default_amplifier(self, embed_dim):
        return nn.Linear(1, embed_dim)

    def default_decoder(self, embed_dim, hidden_dim, dropout, class_num):
        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, class_num)
        )

    def forward(self):
        raise NotImplementedError

    def _amplify(self, x, amplifier):
        x = x.unsqueeze(-1)
        y = amplifier(x)
        return y

    def _encode(self):
        raise NotImplementedError

    def _decode(self, x, decoder):
        y = decoder(x)
        y = F.log_softmax(y, dim=-1)
        return y

    def loss(self, pred, true):
        return self.criterion(pred, true)


# ----------------------RNNs----------------------

class BaseRNNModel(BaseSeqModel):
    def __init__(
        self,
        ENCODER: nn.Module,
        embed_dim: int,
        rnn_layer_num: int,
        class_num: int,
        hidden_dim: int,
        dropout: float,
        *args,
        **kwargs
    ):
        super().__init__()

        self.demand_amplifier = self.default_amplifier(embed_dim)
        self.supply_amplifier = self.default_amplifier(embed_dim)
        self.demand_encoder = ENCODER(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=rnn_layer_num,
            batch_first=True
        )
        self.supply_encoder = ENCODER(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=rnn_layer_num,
            batch_first=True
        )
        self.demand_decoder = self.default_decoder(embed_dim, hidden_dim, dropout, class_num)
        self.supply_decoder = self.default_decoder(embed_dim, hidden_dim, dropout, class_num)

    def forward(self, x, l):
        d_x, s_x = x
        d_x = self._amplify(d_x, self.demand_amplifier)
        s_x = self._amplify(s_x, self.supply_amplifier)
        d_x = self._encode(d_x, l, self.demand_encoder)
        s_x = self._encode(s_x, l, self.supply_encoder)
        d_y = self._decode(d_x, self.demand_decoder)
        s_y = self._decode(s_x, self.supply_decoder)
        return d_y, s_y

    def _encode(self, x, l, encoder):
        x = pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = encoder(x)
        y, _ = pad_packed_sequence(y, batch_first=True)
        y = _get_masked_seq_last(y, l)
        return y


class RNN(BaseRNNModel):

    def __init__(self, *args, **kwargs):
        super().__init__(ENCODER=nn.RNN, *args, **kwargs)


class GRU(BaseRNNModel):

    def __init__(self, *args, **kwargs):
        super().__init__(ENCODER=nn.GRU, *args, **kwargs)


class LSTM(BaseRNNModel):

    def __init__(self, *args, **kwargs):
        super().__init__(ENCODER=nn.LSTM, *args, **kwargs)


# ----------------------DH-GEM----------------------

class DH_GEM(BaseSeqModel):

    def __init__(self, hgs, companies_num, positions_num, embed_dim, com_pos_embed_dim, class_num, hidden_dim, nhead, nhid, nlayers, dropout, *args, **kwargs):
        super().__init__()

        self.src_mask = None
        self.embed_dim = embed_dim

        # dy_com_pos_hgnn
        self.dy_com_pos_hgnn = DyComPosHGNN(hgs, companies_num, positions_num, com_pos_embed_dim)

        # amplifier
        self.demand_amplifier = self.default_amplifier(embed_dim)
        self.supply_amplifier = self.default_amplifier(embed_dim)
        self.compos_amplifier = nn.Linear(embed_dim + com_pos_embed_dim * 2, embed_dim)
        # encoder
        self.position_encoder = PositionalEncoding(embed_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, nhead, nhid, dropout), nlayers)
        self.demand_position_encoder = self.position_encoder
        self.supply_position_encoder = self.position_encoder
        self.demand_transformer_encoder = self.transformer_encoder
        self.supply_transformer_encoder = self.transformer_encoder
        # decoder
        self.com_pos_merge = nn.Linear(com_pos_embed_dim * 2, embed_dim)
        self.demand_supply_merge = nn.Linear(embed_dim * 3, embed_dim)
        self.demand_attention = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.supply_attention = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.demand_decoder = nn.Linear(embed_dim, class_num)
        self.supply_decoder = nn.Linear(embed_dim, class_num)
        self.loss = nn.PoissonNLLLoss()

    def forward(self, x, l, c, p, t_s, t_e):
        d_x, s_x = x
        c, p = self.dy_com_pos_hgnn(c, p, t_s, t_e)  # NOTE
        amp_d_x = self._amplify(d_x, c, p, self.demand_amplifier, self.compos_amplifier)
        amp_s_x = self._amplify(s_x, c, p, self.supply_amplifier, self.compos_amplifier)
        enc_d_x = self._encode(amp_d_x, l, self.position_encoder, self.transformer_encoder)
        enc_s_x = self._encode(amp_s_x, l, self.position_encoder, self.transformer_encoder)
        d_y, s_y = self._decode((enc_d_x, enc_s_x), c, p)
        return (d_y, s_y)

    def _amplify(self, x, c, p, amplifier_1, amplifier_2):
        x = x.unsqueeze(-1)
        y = amplifier_1(x)
        y = torch.concat([y, _repeat_seq_dim(c, y.size(1)), _repeat_seq_dim(p, y.size(1))], dim=-1)
        y = amplifier_2(y)
        return y

    def _encode(self, x, l, position_encoder, transformer_encoder, has_mask=True):
        x = x.permute(1, 0, 2)
        # mask
        if has_mask:
            device = x.device
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = _generate_square_subsequent_mask(len(x)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        # encode
        x = x * math.sqrt(self.embed_dim)
        x = position_encoder(x)
        y = transformer_encoder(x, self.src_mask)
        y = _get_masked_seq_last(y, l, batch_first=False)
        return y

    def _decode(self, x, c, p):
        d_x, s_x = x
        c_p = self.com_pos_merge(torch.concat([c, p], dim=-1))
        d_s = self.demand_supply_merge(torch.concat([d_x, s_x, c_p], dim=-1))
        d_y = self.demand_decoder(self.demand_attention(torch.concat([d_x, d_s], dim=-1)))
        s_y = self.supply_decoder(self.supply_attention(torch.concat([s_x, d_s], dim=-1)))
        return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)


class DyComPosHGNN(nn.Module):

    def __init__(self, hgs, companies_num, positions_num, com_pos_embed_dim):
        super().__init__()
        self.n = len(hgs)
        self.hgs = hgs
        self.hgnn = ComPosHGNN(com_pos_embed_dim)
        self.init_com_emb = nn.Embedding(companies_num, com_pos_embed_dim)
        self.init_pos_emb = nn.Embedding(positions_num, com_pos_embed_dim)
        self.init_com_pos_emb = {'Company': self.init_com_emb.weight, 'Position': self.init_pos_emb.weight}
        self.com_embs = []
        self.pos_embs = []
        com_pos_emb = {'Company': self.init_com_emb.weight, 'Position': self.init_pos_emb.weight}
        for hg in hgs:
            com_pos_emb = self.hgnn(hg, com_pos_emb)
            self.com_embs.append(com_pos_emb['Company'])
            self.pos_embs.append(com_pos_emb['Position'])
        # self.com_embs = torch.stack(self.com_embs).permute(1, 2, 0)  # [n_com, dim, n_time]
        # self.pos_embs = torch.stack(self.pos_embs).permute(1, 2, 0)  # [n_pos, dim, n_time]
        # self.attn = nn.Parameter(torch.ones(self.n))
        self.com_embs = torch.stack(self.com_embs)
        self.pos_embs = torch.stack(self.pos_embs)

    def forward(self, c, p, t_s, t_e):
        if self.com_embs.device is not c.device:
            self.com_embs = self.com_embs.to(c.device)
        if self.pos_embs.device is not p.device:
            self.pos_embs = self.pos_embs.to(p.device)
        # com_embs = []
        # pos_embs = []
        # for ci, pi, ts, te in zip(c, p, t_s, t_e):
        #     assert 0 <= ts < te < self.n
        #     mask_i = torch.concat([torch.zeros(ts), torch.ones(te - ts + 1), torch.zeros(self.n - te - 1)]).to(self.attn.device)
        #     masked_attn_i = mask_i * self.attn
        #     masked_attn_i = self.attn
        #     com_embs.append((self.com_embs * masked_attn_i).mean(dim=-1)[ci])
        #     pos_embs.append((self.pos_embs * masked_attn_i).mean(dim=-1)[pi])
        com_embs = []
        pos_embs = []
        for ci, pi, ts, te in zip(c, p, t_s, t_e):
            com_embs.append(self.com_embs[te][ci])
            pos_embs.append(self.pos_embs[te][pi])
        return torch.stack(com_embs).to(c.device), torch.stack(pos_embs).to(p.device)


# ----------------------Utilities----------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _get_masked_seq_last(seq: torch.Tensor, l: torch.Tensor, batch_first: bool = True):
    if batch_first:
        assert seq.size(0) == l.size(0)
    else:
        assert seq.size(1) == l.size(0)
    last = []
    for i, x in enumerate(l - 1):
        y = seq[i][x] if batch_first else seq[x][i]
        last.append(y)
    return torch.stack(last)


def _repeat_seq_dim(seq: torch.Tensor, seq_len: int):
    return seq.unsqueeze(1).repeat(1, seq_len, 1)


class ComPosHGNN(nn.Module):

    def __init__(self, com_pos_embed_dim: int):
        super().__init__()
        self.edge_weight_norm = dgl_nn.EdgeWeightNorm(norm='both')
        self.hgnn = dgl_nn.HeteroGraphConv({
            'Demand': dgl_nn.GraphConv(com_pos_embed_dim, com_pos_embed_dim, norm='none', activation=nn.ReLU()),
            'Supply': dgl_nn.GraphConv(com_pos_embed_dim, com_pos_embed_dim, norm='none', activation=nn.ReLU()),
            'CompanyFlow': dgl_nn.GraphConv(com_pos_embed_dim, com_pos_embed_dim, norm='none', activation=nn.ReLU()),
            'PositionFlow': dgl_nn.GraphConv(com_pos_embed_dim, com_pos_embed_dim, norm='none', activation=nn.ReLU()),
        }, aggregate='mean')

    def _norm_weight_weight(self, graph, edge_type):
        edges = dgl.edge_type_subgraph(graph, [edge_type])
        return self.edge_weight_norm(edges, edges.edges[edge_type].data['val'])

    def forward(self, graph, com_pos_emb):
        self.demand_edge_weight_norm = self._norm_weight_weight(graph, 'Demand')
        self.supply_edge_weight_norm = self._norm_weight_weight(graph, 'Supply')
        self.company_flow_edge_weight_norm = self._norm_weight_weight(graph, 'CompanyFlow')
        self.position_flow_edge_weight_norm = self._norm_weight_weight(graph, 'PositionFlow')
        emb = self.hgnn(graph, com_pos_emb, mod_kwargs={
            'Demand': {'edge_weight': self.demand_edge_weight_norm},
            'Supply': {'edge_weight': self.supply_edge_weight_norm},
            'CompanyFlow': {'edge_weight': self.company_flow_edge_weight_norm},
            'PositionFlow': {'edge_weight': self.position_flow_edge_weight_norm},
        })
        return emb

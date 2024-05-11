import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class RGTLayer(nn.Module):
    def __init__(self, d_model, n_head=1, drop=0.1):
        super(RGTLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.msg_fc = nn.Linear(self.d_model * 4, self.n_head * self.d_model, bias=False)
        #self.msg_fc = nn.Linear(self.d_model, self.n_head * self.d_model, bias=False)

        self.qw = nn.Linear(self.d_model * 2, self.n_head * self.d_model, bias=False)import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.GraphEncoder import RGTEncoder, RGCNEncoder
from models.SequenceEncoder import TransformerEncoder, GRUEncoder
import torch_scatter
from models.ConvTransE import ConvTransE

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

class TemporalTransformerHawkesGraphModel(nn.Module):
    def __init__(self, config, eps=0.2, time_span=24, timestep=0.1, hmax=5):
        super(TemporalTransformerHawkesGraphModel, self).__init__()
        self.config = config
        self.n_ent = config.n_ent
        self.n_rel = config.n_rel
        self.d_model = config.d_model
        self.dropout_rate = config.dropout
        self.transformer_layer_num = config.seqTransformerLayerNum
        self.transformer_head_num = config.seqTransformerHeadNum
        self.PAD_TIME = -1
        self.PAD_ENTITY = self.n_ent - 1

        self.ent_embeds = nn.Embedding(self.n_ent, self.d_model)
        self.rel_embeds = nn.Embedding(self.n_rel, self.d_model)
        #self.graph_encoder = RGTEncoder(self.d_model, self.dropout_rate)
        self.graph_encoder = RGCNEncoder(self.d_model, self.n_rel, self.d_model//2, self.dropout_rate)
        self.seq_encoder = TransformerEncoder(self.d_model, self.d_model, self.transformer_layer_num,
                                              self.transformer_head_num, self.dropout_rate)
        self.linear_inten_layer = nn.Linear(self.d_model * 3, self.d_model, bias=False)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        #self.Softplus = nn.Softplus(beta=0.5)
        self.lp_loss_fn = LabelSmoothingCrossEntropy(eps)
        self.ent_decoder = ConvTransE(self.n_ent,self.d_model, self.dropout_rate, self.dropout_rate, self.dropout_rate)

        nn.init.xavier_uniform_(self.ent_embeds.weight)
        nn.init.xavier_uniform_(self.rel_embeds.weight)
        nn.init.xavier_uniform_(self.linear_inten_layer.weight)
        

    def forward(self, query_entities, query_relations, history_graphs, history_times, batch_node_ids):
        bs, hist_len = history_times.size(0), history_times.size(1)

        history_graphs.ndata['h'] = self.ent_embeds(history_graphs.ndata['id']).view(-1, self.d_model)
        history_graphs.edata['h'] = self.rel_embeds(history_graphs.edata['type']).view(-1, self.d_model)
        history_graphs.edata['qrh'] = self.rel_embeds(history_graphs.edata['query_rel']).view(-1, self.d_model)
        history_graphs.edata['qeh'] = self.ent_embeds(history_graphs.edata['query_ent']).view(-1, self.d_model)
        total_nodes_h = self.graph_encoder(history_graphs)

        query_rel_embeds = self.rel_embeds(query_relations) # [bs, d_model]        
        query_ent_embeds = self.ent_embeds(query_entities)

        history_gh = total_nodes_h[batch_node_ids].reshape(bs, hist_len, -1)
        history_pad_mask = (history_times == -1).unsqueeze(1)
        local_type = history_graphs.ndata['id'].reshape([bs, -1])
        return query_ent_embeds, query_rel_embeds, history_gh, history_pad_mask, total_nodes_h, local_type

    def entity_prediction(self, query_time, query_ent_embeds, query_rel_embeds,
                        history_gh, history_times, history_pad_mask, total_nodes_h, local_type):
        bs, hist_len = history_times.size(0), history_times.size(1)
        seq_query_input = query_rel_embeds.unsqueeze(1)
        seq_query_time = query_time.view(-1, 1)  # [bs, 1]
        
        output = self.seq_encoder(history_gh, history_times, seq_query_input, seq_query_time, history_pad_mask)
        output = output[:, -1, :]

        inten_raw = self.linear_inten_layer(
            self.dropout(torch.cat((query_ent_embeds, output, query_rel_embeds), dim=-1)))  # [bs, d_model]

        #global_intes = inten_raw.mm(self.ent_embeds.weight.transpose(0, 1))  # [bs, ent_num]
        global_intes = self.ent_decoder(query_ent_embeds,query_rel_embeds, output,self.ent_embeds.weight)
        local_h = total_nodes_h.reshape([bs, -1, self.d_model])  # [bs, max_nodes_num * seq_len, d_model]
        local_intes = torch.matmul(inten_raw.unsqueeze(1), local_h.transpose(1, 2))[:, -1, :]  # [bs, max_nodes_num * seq_len]
        intens = F.softmax(torch.cat([global_intes, local_intes], dim=-1),dim=-1)

        global_type = torch.arange(self.n_ent, device=intens.device).unsqueeze(0).repeat(bs, 1)
        ent_type = torch.cat([global_type, local_type], dim=-1)
        return intens, ent_type

    def link_prediction_loss(self, intens, type, answers):
        intens = torch_scatter.scatter(intens, type, dim=-1, reduce="mean")
        loss = self.lp_loss_fn(intens[:, :-1], answers)
        return loss

    def time_prediction_loss(self, estimate_dt, dur_last):
        loss_dt = self.tp_loss_fn(estimate_dt, dur_last)
        return loss_dt

    def ents_score(self, intens, type, local_weight=1.):
        intens[:, self.n_ent:] = intens[:, self.n_ent:] * local_weight
        output = torch_scatter.scatter(intens, type, dim=-1, reduce="max")
        return output[:, :-1]


    def train_forward(self, s_ent, relation, o_ent, time, history_graphs, history_times, batch_node_ids):
        query_ent_embeds, query_rel_embeds, history_gh, history_pad_mask, total_nodes_h, local_type = \
            self.forward(s_ent, relation, history_graphs, history_times, batch_node_ids)
        ent_intes, ent_type = self.entity_prediction(time, query_ent_embeds, query_rel_embeds, history_gh, history_times, history_pad_mask,
                                                total_nodes_h, local_type)
        loss_lp = self.link_prediction_loss(ent_intes, ent_type, o_ent)
        
        return loss_lp
    def test_forward(self, s_ent, relation, o_ent, time, history_graphs, history_times, batch_node_ids, local_weight=1.):
        query_ent_embeds, query_rel_embeds, history_gh, history_pad_mask, total_nodes_h, local_type = \
            self.forward(s_ent, relation, history_graphs, history_times, batch_node_ids)

        type_intes, ent_type = self.entity_prediction(time, query_ent_embeds, query_rel_embeds, history_gh, history_times,
                                                history_pad_mask,
                                                total_nodes_h, local_type)
        scores = self.ents_score(type_intes, ent_type, local_weight)

        return scores
        self.kw = nn.Linear(self.d_model * 2, self.n_head * self.d_model, bias=False)
        self.temp = self.d_model ** 0.5

        self.output_fc = nn.Linear(self.n_head * self.d_model, self.d_model, bias=False)

        self.layer_norm1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.dropout = nn.Dropout(drop)

        nn.init.xavier_uniform_(self.qw.weight)
        nn.init.xavier_uniform_(self.kw.weight)
        nn.init.xavier_uniform_(self.msg_fc.weight)

    def msg_func(self, edges):
        msg = self.msg_fc(torch.cat([edges.src['h'], edges.data['h'],edges.data['qrh'],edges.data['qeh']], dim=-1))
        #msg = self.msg_fc(edges.data['h'])
        msg = F.leaky_relu(msg)
        msg = self.dropout(msg)
        q = self.qw(torch.cat([edges.data['qrh'],edges.data['qeh']],dim=-1)) / self.temp
        k = self.kw(torch.cat([edges.src['h'], edges.data['h']], dim=-1))
        #k = self.kw(edges.data['h'])
        msg = msg.view(-1, self.n_head, self.d_model)
        q = q.view(-1, self.n_head, self.d_model)
        k = k.view(-1, self.n_head, self.d_model)
        att = torch.sum(q * k, dim=-1).unsqueeze(-1)  # [-1, n_head, 1]
        return {'msg': msg, 'att': att}

    def reduce_func(self, nodes):
        res = nodes.data['h']
        alpha = self.dropout(F.softmax(nodes.mailbox['att'], dim=1))
        h = torch.sum(alpha * nodes.mailbox['msg'], dim=1).view(-1, self.n_head * self.d_model)
        h = self.dropout(F.leaky_relu(self.output_fc(h)))
        # print(nodes.data['type'])
        # print(alpha.squeeze(-1))

        h = h + res
        h = self.layer_norm1(h)
        return {'h': h}

    def forward(self, g):
        g.update_all(self.msg_func, self.reduce_func)
        return g

class RGTEncoder(nn.Module):
    def __init__(self, d_model, drop=0.1, n_head=1):
        super(RGTEncoder, self).__init__()
        self.layer1 = RGTLayer(d_model, n_head, drop)
        self.layer2 = RGTLayer(d_model, n_head, drop)
        # self.layer3 = RGTLayer(d_model, n_head, drop)
        # self.layer4 = RGTLayer(d_model, n_head, drop)

    def forward(self, g):
        self.layer1(g)
        self.layer2(g)
        # self.layer3(g)
        # self.layer4(g)
        return g.ndata['h']


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=False,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.self_loop = self_loop
        self.activation = activation
        self.out_feat = out_feat
        self.in_feat = in_feat

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        self.fc = nn.Linear(self.in_feat * 4, self.in_feat, bias=False)

        if self.bias:
            self.bias_parm = nn.Parameter(torch.zeros(out_feat))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(-1, self.submat_in, self.submat_out)
        node = self.fc(torch.cat([edges.src['h'], edges.data['h'],edges.data['qrh'],edges.data['qeh']], dim=-1)).view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h']}

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            loop_message = self.dropout(loop_message)

        self.propagate(g)

        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias_parm
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = F.relu(node_repr)
        return g

class RGCNEncoder(nn.Module):
    def __init__(self, ent_dim, num_rels, num_bases, dropout=0.0):
        super(RGCNEncoder, self).__init__()
        self.layer1 = RGCNLayer(ent_dim, ent_dim, num_rels, num_bases, True, torch.nn.functional.relu, True, dropout)
        self.layer2 = RGCNLayer(ent_dim, ent_dim, num_rels, num_bases, True, None, True, dropout)

    def forward(self, g):
        self.layer1(g)
        self.layer2(g)
        return g.ndata['h']

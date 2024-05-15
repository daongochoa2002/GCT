import random
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
        self.graph_encoder = RGTEncoder(self.d_model, self.dropout_rate)
        #self.graph_encoder = RGCNEncoder(self.d_model, self.n_rel, self.d_model//2, self.dropout_rate)
        self.seq_encoder = TransformerEncoder(self.d_model, self.d_model, self.transformer_layer_num,
                                              self.transformer_head_num, self.dropout_rate)
        self.linear_inten_layer = nn.Linear(self.d_model * 3, self.d_model, bias=False)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.Softplus = nn.Softplus(beta=10)
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
        global_type = torch.arange(self.n_ent, device=output.device).unsqueeze(0).repeat(bs, 1)
        global_intes = self.ent_decoder(query_ent_embeds,query_rel_embeds, output,self.ent_embeds.weight)
        global_intes *= (global_type != self.PAD_ENTITY) + 0.2
        non_history = torch.zeros_like(global_intes)
        local_h = total_nodes_h.reshape([bs, -1, self.d_model])  # [bs, max_nodes_num * seq_len, d_model]
        local_intes = torch.matmul(inten_raw.unsqueeze(1), local_h.transpose(1, 2))[:, -1, :]  # [bs, max_nodes_num * seq_len]
        history =  torch.ones_like(local_intes)
        history_tag = torch.cat([non_history, history], dim=-1)
        local_intes *= (local_type != self.PAD_ENTITY) + 0.2
        intens = self.Softplus(torch.cat([global_intes, local_intes], dim=-1) + history_tag)

        
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
        intens = F.softmax(intens, dim=-1)
        intens[:, self.n_ent:] = intens[:, self.n_ent:] * 0.8
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

import torch
import torch.nn.functional as F
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, gcn_layers, alpha, droprate):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.weights_pool2 = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_out, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.gcn_layer = gcn_layers
        self.alpha = alpha
        self.droprate = droprate
    def forward(self, x, node_embeddings):
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        x_gconv0 = x_gconv
        for i in range(self.gcn_layer - 1):
            x = x_gconv
            weights2 = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool2)  # N, cheb_k, dim_in, dim_out
            bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
            # drop
            drop = F.relu(torch.rand((node_num, node_num), out=None) - self.droprate * torch.ones(node_num, node_num))
            drop = drop.to(supports.device)
            supports = torch.mul(supports, drop)
            x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
            x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
            x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights2) + bias + x_gconv0

        return x_gconv
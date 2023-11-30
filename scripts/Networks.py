import os
import torch
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_scatter import scatter_max, scatter_add
from torch_geometric.nn import GCNConv, TopKPooling, GatedGraphConv, global_max_pool, global_mean_pool
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops, softmax)
from torch_geometric.utils.repeat import repeat


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(GCN, self).__init__()
        assert num_layers >= 1
        self.covs = torch.nn.ModuleList()
        self.covs.append(GCNConv(in_channels, hidden_channels, improved=True))
        for _ in range(num_layers - 1):
            self.covs.append(
                GCNConv(hidden_channels, hidden_channels, improved=True))
        self.fully_con1 = torch.nn.Linear(hidden_channels, 1)

        # self.conv1 = GCNConv(in_channels, hidden_channels, improved=True)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, improved=True)

    def forward(self, data, prob, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for conv in self.covs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        # x = self.conv1(x, edge_index, edge_weight=edge_weight)
        # x = F.relu(x)
        # x = self.conv2(x, edge_index, edge_weight=edge_weight)
        # x = F.relu(x)
        x = F.dropout(x, p=prob)
        x = self.fully_con1(x)
        return x


class PolicyGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(PolicyGCN, self).__init__()

        assert num_layers >= 1
        self.covs = torch.nn.ModuleList()
        self.covs.append(GCNConv(in_channels, hidden_channels, improved=True))
        for _ in range(num_layers - 1):
            self.covs.append(
                GCNConv(hidden_channels, hidden_channels, improved=True))
        self.fully_con1 = torch.nn.Linear(hidden_channels, 1)

        # self.conv1 = GCNConv(in_channels, hidden_channels, improved=True)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, improved=True)

    def forward(self, data, mask, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        for conv in self.covs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))

        x = F.dropout(x)
        x = self.fully_con1(x)

        # x = self.conv1(x, edge_index, edge_weight=edge_weight)
        # x = F.relu(x)
        # x = self.conv2(x, edge_index, edge_weight=edge_weight)
        # x = F.relu(x)
        # x = F.dropout(x)
        # x = self.fully_con1(x)

        # 只保留actions(frontiers)部分
        x = torch.masked_select(x.view(-1), mask)
        batch = torch.masked_select(batch, mask)
        x = softmax(x, batch)
        return x


class ValueGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(ValueGCN, self).__init__()

        assert num_layers >= 1
        self.covs = torch.nn.ModuleList()
        self.covs.append(GCNConv(in_channels, hidden_channels, improved=True))
        for _ in range(num_layers - 1):
            self.covs.append(
                GCNConv(hidden_channels, hidden_channels, improved=True))
        self.fully_con1 = torch.nn.Linear(hidden_channels, 100)

        # self.conv1 = GCNConv(in_channels, hidden_channels, improved=True)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, improved=True)

    def forward(self, data, mask, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for conv in self.covs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        # x = self.conv1(x, edge_index, edge_weight=edge_weight)
        # x = F.relu(x)
        # x = self.conv2(x, edge_index, edge_weight=edge_weight)
        # x = F.relu(x)
        x = F.dropout(x)
        x = self.fully_con1(x)
        x = global_mean_pool(x, batch).mean(dim=1)
        return x


class GGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(GGNN, self).__init__()
        self.gconv1 = GatedGraphConv(hidden_channels, num_layers)
        self.fully_con1 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data, prob, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.gconv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=prob)
        x = self.fully_con1(x)
        return x


class PolicyGGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(PolicyGGNN, self).__init__()
        self.gconv1 = GatedGraphConv(hidden_channels, num_layers)
        self.fully_con1 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data, mask, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.gconv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fully_con1(x)
        x = torch.masked_select(x.view(-1), mask)
        batch = torch.masked_select(batch, mask)
        x = softmax(x, batch)
        return x


class ValueGGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(ValueGGNN, self).__init__()
        self.gconv1 = GatedGraphConv(hidden_channels, num_layers)
        self.fully_con1 = torch.nn.Linear(hidden_channels, 100)

    def forward(self, data, mask, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.gconv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fully_con1(x)
        x = global_mean_pool(x, batch).mean(dim=1)
        return x


class GraphUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(GraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))
        self.fully_con1 = torch.nn.Linear(self.out_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, data, prob, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        x = F.relu(x)
        x = F.dropout(x, p=prob)
        x = self.fully_con1(x)

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)


class PolicyGraphUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(PolicyGraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))
        self.fully_con1 = torch.nn.Linear(self.out_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, data, mask, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        old_batch = batch.clone()

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        x = F.relu(x)
        x = F.dropout(x)
        x = self.fully_con1(x)
        x = torch.masked_select(x.view(-1), mask)
        old_batch = torch.masked_select(old_batch, mask)
        x = softmax(x, old_batch)

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)


class ValueGraphUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(ValueGraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))
        self.fully_con1 = torch.nn.Linear(self.out_channels, 100)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, data, mask, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # edge_weight = x.new_ones(edge_index.size(1))
        old_batch = batch.clone()
        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        x = F.relu(x)
        x = F.dropout(x)
        x = self.fully_con1(x)
        x = global_mean_pool(x, old_batch).mean(dim=1)
        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)

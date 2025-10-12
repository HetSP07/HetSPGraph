import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *

class gtnet(nn.Module):
    """
    Graph Temporal Network for heterogeneous graphs.
    Supports multiple adjacency matrices passed as a dictionary during forward pass.
    """
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, 
                 predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, 
                 dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, 
                 end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, 
                 propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential**layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        new_dilation = 1
        for i in range(layers):
            self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
            self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
            self.residual_convs.append(nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, seq_length)))

            if self.gcn_true:
                self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

            self.norm.append(LayerNorm((residual_channels, num_nodes, seq_length), elementwise_affine=layer_norm_affline))
            new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1,1))
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1,1))

        self.skip0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, seq_length))
        self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1))

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, adjs=None, idx=None):
        """
        Forward pass:
        - input: (batch, features, num_nodes, seq_len)
        - adjs: dictionary of adjacency matrices for heterogeneous graph
        - idx: node indices (optional)
        """
        seq_len = input.size(3)
        if seq_len != self.seq_length:
            if seq_len < self.receptive_field:
                input = F.pad(input, (self.receptive_field - seq_len, 0, 0, 0))
            else:
                raise ValueError(f"Input sequence length {seq_len} != model sequence length {self.seq_length}")

        # Build adaptive adjacency matrices if needed
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)

            s = self.skip_convs[i](x)
            skip = skip + s

            # Apply heterogeneous graph convolution if provided
            if self.gcn_true:
                if adjs is not None:
                    # sum over all adjacency types
                    x_hetero = 0
                    for key, adj in adjs.items():
                        x_hetero += self.gconv1[i](x, adj) + self.gconv2[i](x, adj.transpose(1,0))
                    x = x_hetero
                else:
                    x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

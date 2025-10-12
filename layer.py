# hetero_layers.py
from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numbers

################################################################################
# Basic conv / linear layers (unchanged/ported)
################################################################################

class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        # 1x1 conv over (channels, nodes, time) layout used in original code
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


################################################################################
# Heterogeneous nconv: combine multiple relation adjacency matrices
################################################################################

class hetero_nconv(nn.Module):
    """
    Heterogeneous neighborhood convolution.
    - x: tensor with shape (N_batch? or N?, C, V, L) using same layout as original code: 'ncwl'
    - A_dict: dict of relation_name -> adjacency matrix (V,V) or (B, V, V) for batched adj
    combine_mode: 'sum' | 'concat' | 'attention'
    """
    def __init__(self, relations, combine_mode='sum', in_channels=None):
        super(hetero_nconv, self).__init__()
        assert combine_mode in ('sum', 'concat', 'attention')
        self.relations = list(relations)
        self.combine_mode = combine_mode
        self.in_channels = in_channels
        if combine_mode == 'attention':
            # attention across relations: per-relation scalar score learned from node features
            # We'll use a small MLP to compute attention logits per relation, shared across nodes
            # Input channel dimension is in_channels (C), but if None we will infer later lazily.
            assert in_channels is not None, "in_channels must be provided for attention combine_mode"
            self.att_mlp = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, len(self.relations), kernel_size=1)  # outputs per-relation logits
            )

    def forward(self, x, A_dict):
        """
        x: (batch?, C, V, L) or (C, V, L) -- keep same ops as original with einsum
        A_dict: {rel: A} where A is (V,V) or (B,V,V) (if batched)
        returns: combined output with same shape as a single relation conv when combine_mode='sum' or 'attention';
                 when 'concat', channel dim grows by len(relations)
        """
        rel_outs = []
        for rel in self.relations:
            A = A_dict[rel]
            # support (V,V) or (B,V,V). Original einsum 'ncwl,vw->ncvl' expects 'vw' shape (V,V)
            # If A is batched (B,V,V) and x is batched we handle separately.
            if A.dim() == 2:
                out = torch.einsum('ncwl,vw->ncvl', (x, A))
            elif A.dim() == 3:
                # Assume batch dimension aligns with x's batch dimension if present
                # Try: x is (B, C, V, L) and A is (B, V, V) -> use bmm per channel/time slice
                B, C, V, L = x.shape
                out = torch.zeros_like(x)  # (B,C,V,L)
                # compute for each time slice to match einsum behaviour across last dim L
                for t in range(L):
                    xt = x[..., t]  # (B,C,V)
                    # compute: for each batch b and each channel c: (V) <- A[b] @ xt[b,c,:].T
                    # efficient way: reshape and bmm
                    xt_ = xt.permute(0, 2, 1)  # (B, V, C)
                    # compute A @ xt_ -> (B, V, C)
                    y = torch.bmm(A, xt_)  # (B,V,C)
                    y = y.permute(0, 2, 1).unsqueeze(-1)  # (B,C,V,1)
                    out[..., t:t+1] = y
            else:
                raise ValueError("Adjacency matrix A must be 2D or 3D (batched).")
            rel_outs.append(out)

        if self.combine_mode == 'sum':
            return torch.stack(rel_outs, dim=0).sum(dim=0)  # sum across relations
        elif self.combine_mode == 'concat':
            return torch.cat(rel_outs, dim=1)  # concat on channel axis
        else:  # attention
            # att_mlp expects same spatial dims; produce per-relation logits and softmax across relations
            # att_logits shape: (B or 1, R, V, L) -> softmax over R
            att_logits = self.att_mlp(x)  # (B, R, V, L)
            att_scores = torch.softmax(att_logits, dim=1)  # across relations
            # weighted sum:
            out = 0
            for i, rel in enumerate(self.relations):
                out = out + rel_outs[i] * att_scores[:, i:i+1, :, :]
            return out


################################################################################
# Dynamic hetero nconv - compute relation adj dynamically from node features
################################################################################

class hetero_dy_nconv(nn.Module):
    """
    Dynamic adjacency computed from node embeddings per relation.
    We'll compute for each relation either:
      - A_rel = softmax( nodevec1 @ nodevec2^T )  (like dy_mixprop original)
    Input x: (B, C, V, L)
    """
    def __init__(self, relations, in_channels, hidden_dim=None, combine_mode='sum', alpha=3.0, device='cpu'):
        super(hetero_dy_nconv, self).__init__()
        self.relations = list(relations)
        self.combine_mode = combine_mode
        self.alpha = alpha
        self.device = device
        self.in_channels = in_channels
        if hidden_dim is None:
            hidden_dim = in_channels
        # per relation pair of linear layers to produce node vectors (we keep them shared but separate could be used)
        self.lin1 = nn.ModuleDict({r: nn.Linear(in_channels, hidden_dim) for r in self.relations})
        self.lin2 = nn.ModuleDict({r: nn.Linear(in_channels, hidden_dim) for r in self.relations})
        # reuse hetero_nconv to apply conv once adjacency computed
        self.base_nconv = hetero_nconv(relations=self.relations, combine_mode=combine_mode, in_channels=in_channels)

    def forward(self, x):
        # x: (B, C, V, L)
        B, C, V, L = x.shape
        # build A_dict dynamically per relation (shape (B,V,V))
        A_dict = {}
        # create node embeddings from temporal-aggregated features: e.g., mean over time and channels
        # We'll use mean over channels and time to create a node embedding (B,V,C)->(B,V)
        node_feat = x.mean(dim=1).permute(0, 2, 1)  # (B, V, L) -> but we want a single vector per node -> mean over time
        node_vec = node_feat.mean(dim=-1)  # (B, V)
        # To preserve dimensionality, expand to C
        node_vec_expanded = node_vec.unsqueeze(-1).repeat(1, 1, C)  # (B, V, C)
        # But lin layers expect last dim = in_channels; permute
        node_vec_for_lin = node_vec_expanded  # (B, V, C)
        for r in self.relations:
            lin1 = self.lin1[r]
            lin2 = self.lin2[r]
            # apply per-batch node transforms
            # flatten batch+nodes to transform with linear
            nv = node_vec_for_lin.view(B * V, C)
            v1 = torch.tanh(self.alpha * lin1(nv)).view(B, V, -1)  # (B, V, H)
            v2 = torch.tanh(self.alpha * lin2(nv)).view(B, V, -1)  # (B, V, H)
            # compute adjacency per batch: A[b] = softmax( v1[b] @ v2[b].T )
            A_batch = []
            for b in range(B):
                a = torch.matmul(v1[b], v2[b].transpose(0, 1))  # (V,V)
                a = F.relu(torch.tanh(self.alpha * a))
                # row-softmax
                A_batch.append(F.softmax(a, dim=1))
            A_batch = torch.stack(A_batch, dim=0)  # (B, V, V)
            A_dict[r] = A_batch
        # Now use base nconv (which supports batched A) to convolve
        return self.base_nconv(x, A_dict)


################################################################################
# Propagation and MixProp adapted to heterogeneous adjacency dictionaries
################################################################################

class hetero_prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha, relations, combine_mode='sum', in_channels=None):
        super(hetero_prop, self).__init__()
        self.nconv = hetero_nconv(relations=relations, combine_mode=combine_mode, in_channels=in_channels)
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, A_dict):
        # A_dict: dict of adjacency matrices; we will sum them for normalization as earlier code expects single adj
        # Add self-loop to each adjacency matrix
        # Here x shape: (B?, C, V, L) or (C, V, L) consistent with nconv
        # For simplicity, we produce an aggregated adjacency for normalization use
        # create aggregated adj: sum of relations
        any_rel = next(iter(A_dict.values()))
        if any_rel.dim() == 2:
            agg = torch.zeros_like(any_rel)
        else:
            agg = torch.zeros_like(any_rel)
        for A in A_dict.values():
            agg = agg + A
        # self loop
        if agg.dim() == 2:
            agg = agg + torch.eye(agg.size(0)).to(x.device)
            d = agg.sum(1)
            a_norm = agg / d.view(-1, 1)
        else:
            B, V, _ = agg.shape
            I = torch.eye(V).to(x.device).unsqueeze(0).repeat(B, 1, 1)
            agg = agg + I
            d = agg.sum(2)  # (B, V)
            a_norm = agg / (d.unsqueeze(-1) + 1e-8)

        h = x
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, A_dict)
        ho = self.mlp(h)
        return ho


class hetero_mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha, relations, combine_mode='sum', in_channels=None):
        super(hetero_mixprop, self).__init__()
        self.nconv = hetero_nconv(relations=relations, combine_mode=combine_mode, in_channels=in_channels)
        # If combine_mode == 'concat', nconv returns increased channels: account for that
        if combine_mode == 'concat':
            eff_in = c_in * (gdep + 1) * len(relations)
        else:
            eff_in = c_in * (gdep + 1)
        self.mlp = linear(eff_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, A_dict):
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, A_dict)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


################################################################################
# Dynamic mixprop (dy_mixprop) adapted: computes dynamic adj per relation from node features
################################################################################

class hetero_dy_mixprop(nn.Module):
    """
    Adapts original dy_mixprop to hetero setup:
    - computes dynamic adjacency per relation from learned linear projections of x
    - applies propagation per relation direction and sums results
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha, relations, in_channels=None):
        super(hetero_dy_mixprop, self).__init__()
        self.relations = relations
        self.gdep = gdep
        self.alpha = alpha
        self.lin1 = nn.ModuleDict({r: linear(c_in, c_in) for r in relations})
        self.lin2 = nn.ModuleDict({r: linear(c_in, c_in) for r in relations})
        # two mlps as in original dy_mixprop; keep one per relation or shared; here we use shared
        if isinstance(c_out, int):
            self.mlp1 = linear((gdep + 1) * c_in, c_out)
            self.mlp2 = linear((gdep + 1) * c_in, c_out)
        else:
            self.mlp1 = linear((gdep + 1) * c_in, c_out)
            self.mlp2 = linear((gdep + 1) * c_in, c_out)

    def forward(self, x):
        # x: (B, C, V, L)
        B, C, V, L = x.shape
        # compute dynamic A per relation using linear transforms on features (using last time slice features)
        # we take last time slice as node embedding base as in many temporal GNNs
        x_last = x[..., -1]  # (B, C, V)
        x_last = x_last.permute(0, 2, 1)  # (B, V, C)
        A_dict = {}
        for r in self.relations:
            v1 = torch.tanh(self.lin1[r](x_last.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1))
            # simplify: use linear over channel dimension by flatten/unflatten
            # fallback simpler approach: compute with small MLP: flatten last dim
            v1 = torch.tanh(self.lin1[r](x_last.reshape(B * V, C))).view(B, V, C)
            v2 = torch.tanh(self.lin2[r](x_last.reshape(B * V, C))).view(B, V, C)
            A_batch = []
            for b in range(B):
                a = torch.matmul(v1[b], v2[b].transpose(0, 1))  # (V,V)
                a = F.relu(torch.tanh(self.alpha * a))
                A_batch.append(F.softmax(a, dim=1))
            A_batch = torch.stack(A_batch, dim=0)  # (B, V, V)
            A_dict[r] = A_batch

        # apply propagation similar to original dy_mixprop but summing relation contributions
        # first path (using adj as computed)
        h = x
        out = [h]
        for i in range(self.gdep):
            # sum over relation convs
            rel_sum = None
            for r, A in A_dict.items():
                conv = torch.einsum('ncvl,nvwl->ncwl', (h, A))  # dynamic conv for batched A
                rel_sum = conv if rel_sum is None else rel_sum + conv
            h = self.alpha * x + (1 - self.alpha) * rel_sum
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        # second path: transpose adjacency (reverse direction) and repeat
        h = x
        out = [h]
        for i in range(self.gdep):
            rel_sum = None
            for r, A in A_dict.items():
                A_t = A.transpose(1, 2)  # (B,V,V)
                conv = torch.einsum('ncvl,nvwl->ncwl', (h, A_t))
                rel_sum = conv if rel_sum is None else rel_sum + conv
            h = self.alpha * x + (1 - self.alpha) * rel_sum
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1 + ho2


################################################################################
# Temporal convolutions (unchanged)
################################################################################

class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, input):
        x = self.tconv(input)
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        self.tconv = nn.ModuleList([nn.Conv2d(cin, cout, (1, k), dilation=(1, dilation_factor)) for k in self.kernel_set])

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        # align time dimension to the smallest produced by kernels
        for i in range(len(x)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


################################################################################
# Graph constructors producing multiple adjacency matrices (one per relation)
################################################################################

class graph_constructor_hetero(nn.Module):
    """
    Construct multiple adjacency matrices for different relations.
    relation_specs: dict where keys are relation names and values are spec dicts, for example:
      {"road": {"type":"directed", "k":8, "dim":16},
       "sim": {"type":"undirected", "k":16, "dim":16},
      }
    static_feat: optional (V, feat_dim) tensor used for static node features
    """
    def __init__(self, nnodes, relation_specs, device='cpu', alpha=3.0, static_feat=None):
        super(graph_constructor_hetero, self).__init__()
        self.nnodes = nnodes
        self.relation_specs = relation_specs
        self.device = device
        self.alpha = alpha
        self.static_feat = static_feat
        # per relation emb/lin depending on whether static_feat provided
        self.rel_modules = nn.ModuleDict()
        for rel, spec in relation_specs.items():
            dim = spec.get("dim", 16)
            if static_feat is None:
                emb1 = nn.Embedding(nnodes, dim)
                emb2 = nn.Embedding(nnodes, dim) if spec.get("type", "directed") == "directed" else None
                lin1 = nn.Linear(dim, dim)
                lin2 = nn.Linear(dim, dim) if spec.get("type", "directed") == "directed" else lin1
                self.rel_modules[rel] = nn.ModuleDict({
                    "emb1": emb1,
                    "emb2": emb2 if emb2 is not None else emb1,
                    "lin1": lin1,
                    "lin2": lin2
                })
            else:
                xd = static_feat.shape[1]
                lin1 = nn.Linear(xd, dim)
                lin2 = nn.Linear(xd, dim) if spec.get("type", "directed") == "directed" else lin1
                self.rel_modules[rel] = nn.ModuleDict({
                    "lin1": lin1,
                    "lin2": lin2
                })
        self.static_feat = static_feat

    def forward(self, idx):
        """
        idx: tensor of node indices (V,) or (B, V) to construct per-batch adjacency
        returns: dict rel -> adjacency (V,V) or (B, V, V) depending on idx dim
        """
        device = self.device
        out = {}
        batched = (idx.dim() == 2)
        if not batched:
            V = idx.size(0)
            for rel, spec in self.relation_specs.items():
                mod = self.rel_modules[rel]
                if self.static_feat is None:
                    nodevec1 = mod["emb1"](idx)
                    nodevec2 = mod["emb2"](idx)
                else:
                    nodevec1 = self.static_feat[idx, :]
                    nodevec2 = nodevec1
                nodevec1 = torch.tanh(self.alpha * mod["lin1"](nodevec1))
                nodevec2 = torch.tanh(self.alpha * mod["lin2"](nodevec2))
                a = torch.mm(nodevec1, nodevec2.t()) - torch.mm(nodevec2, nodevec1.t())
                adj = F.relu(torch.tanh(self.alpha * a))
                # sparsify top-k if k in spec
                k = spec.get("k", None)
                if k is not None:
                    mask = torch.zeros(V, V).to(device)
                    s1, t1 = (adj + torch.rand_like(adj) * 1e-4).topk(k, 1)
                    mask.scatter_(1, t1, s1.fill_(1.0))
                    adj = adj * mask
                out[rel] = adj
        else:
            B = idx.size(0)
            V = idx.size(1)
            for rel, spec in self.relation_specs.items():
                mod = self.rel_modules[rel]
                A_batches = []
                for b in range(B):
                    idxb = idx[b]
                    if self.static_feat is None:
                        nodevec1 = mod["emb1"](idxb)
                        nodevec2 = mod["emb2"](idxb)
                    else:
                        nodevec1 = self.static_feat[idxb, :]
                        nodevec2 = nodevec1
                    nodevec1 = torch.tanh(self.alpha * mod["lin1"](nodevec1))
                    nodevec2 = torch.tanh(self.alpha * mod["lin2"](nodevec2))
                    a = torch.mm(nodevec1, nodevec2.t()) - torch.mm(nodevec2, nodevec1.t())
                    adj = F.relu(torch.tanh(self.alpha * a))
                    k = spec.get("k", None)
                    if k is not None:
                        mask = torch.zeros(V, V).to(device)
                        s1, t1 = (adj + torch.rand_like(adj) * 1e-4).topk(k, 1)
                        mask.scatter_(1, t1, s1.fill_(1.0))
                        adj = adj * mask
                    A_batches.append(adj)
                out[rel] = torch.stack(A_batches, dim=0)  # (B, V, V)
        return out


################################################################################
# Simple global adjacency (shared) per relation (learnable)
################################################################################

class graph_global_hetero(nn.Module):
    def __init__(self, nnodes, relations, device='cpu'):
        super(graph_global_hetero, self).__init__()
        self.nnodes = nnodes
        self.relations = relations
        self.device = device
        self.A_params = nn.ParameterDict({r: nn.Parameter(torch.randn(nnodes, nnodes).to(device)) for r in relations})

    def forward(self, idx=None):
        # return dict of rel -> adj (V,V)
        out = {}
        for r, p in self.A_params.items():
            out[r] = F.relu(p)
        return out


################################################################################
# LayerNorm (kept similar but slightly simplified for hetero)
################################################################################

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx=None):
        # idx is ignored here for simplicity; if you need node-specific params, supply weight shaped accordingly
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), None, None, self.eps)

################################################################################
# End of module
################################################################################

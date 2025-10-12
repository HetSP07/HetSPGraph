import torch
from torch_geometric.data import HeteroData

def generate_hetero_graph_seq2seq_data(
    df, x_offsets, y_offsets, edge_index_dict, add_time_in_day=True, add_day_in_week=False
):
    num_samples, num_nodes = df.shape
    x_graphs, y_data = [], []
    
    data = torch.tensor(df.values, dtype=torch.float).unsqueeze(-1)  # (T, N, 1)

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = torch.tensor(time_ind, dtype=torch.float).unsqueeze(-1).repeat(1, num_nodes, 1)
        data = torch.cat([data, time_in_day], dim=-1)

    if add_day_in_week:
        day_in_week = torch.zeros((num_samples, num_nodes, 7))
        day_in_week[torch.arange(num_samples), :, df.index.dayofweek] = 1
        data = torch.cat([data, day_in_week], dim=-1)

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    for t in range(min_t, max_t):
        x_window = data[t + x_offsets]  # (input_len, num_nodes, channels)
        
        graph = HeteroData()
        graph["sensor"].x = x_window[-1]  # last state as node features
        
        for rel, edge_index in edge_index_dict.items():
            src, dst = rel.split("::")
            graph[(src, "to", dst)].edge_index = torch.tensor(edge_index, dtype=torch.long)

        x_graphs.append(graph)

        y_window = data[t + y_offsets]  # (out_len, num_nodes, channels)
        y_data.append(y_window)

    return x_graphs, torch.stack(y_data)

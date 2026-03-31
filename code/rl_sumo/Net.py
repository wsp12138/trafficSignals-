import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GCNConv,GATConv,TopKPooling



class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.input_dim, args.hidden_dim)
        self.conv2 = GCNConv(args.hidden_dim, args.output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

#
class GCN_(nn.Module):
    def __init__(self, args):
        super(GCN_, self).__init__()

        self.conv1 = GCNConv(args.input_dim, args.hidden_dim)
        self.conv2 = GCNConv(args.hidden_dim, args.hidden_dim)
        self.fc1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class effect_net(nn.Module):
    def __init__(self, args):
        super(effect_net, self).__init__()
        self.f1 = nn.Linear(args.effect_input_dim, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim, args.effect_output_dim)

    def forward(self, x):
        x = F.relu(self.f1(x))

        return self.f2(x)

class NodeEncoder(nn.Module):
    def __init__(self, in_dim,embed_dim):
        super(NodeEncoder, self).__init__()
        self.fc = nn.Linear(in_dim, embed_dim)

    def forward(self,x):
        return F.relu(self.fc(x))


class NodeDecoder(nn.Module):
    def __init__(self, hidden_dim,output_dim):
        super(NodeDecoder, self).__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class GCN_for_complex(nn.Module):
    def __init__(self, args):
        super(GCN_for_complex, self).__init__()
        self.num_nodes = len(args.inputs_dim)
        self.inputs_dim = args.inputs_dim
        self.outputs_dim = args.outputs_dim
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim

        self.encoders = nn.ModuleList([NodeEncoder(in_dim,self.embed_dim) for in_dim in self.inputs_dim])
        self.conv1 = GCNConv(self.embed_dim, self.hidden_dim)
        # self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)

        self.decoders = nn.ModuleList([NodeDecoder(self.hidden_dim, output_dim) for output_dim in self.outputs_dim])

    def forward(self, x, edge_index):
        if isinstance(x, list) and x[0].dim() == 2:
            batch_size = x[0].size(0)
            encoded_nodes = []
            for i in range(self.num_nodes):
                # 编码：每个节点独立映射 -> [batch_size, embed_dim]
                encoded = self.encoders[i](x[i])
                encoded_nodes.append(encoded)
            # 合并后 shape: [batch_size, num_nodes, embed_dim]
            encoded_stack = torch.stack(encoded_nodes, dim=1)
            # 针对每个样本分别进行图卷积（图结构相同）
            out_list = []
            for b in range(batch_size):
                sample = encoded_stack[b]  # [num_nodes, embed_dim]
                sample_gc = F.relu(self.conv1(sample, edge_index))
                # sample_gc = F.relu(self.conv2(sample_gc, edge_index))


                out_list.append(sample_gc)
            x_gc = torch.stack(out_list, dim=0)  # [batch_size, num_nodes, hidden_dim]
            outputs = []
            for i in range(self.num_nodes):
                # 对于节点 i，解码得到 Q 值，形状为 [batch_size, out_dim_i]
                q_values = self.decoders[i](x_gc[:, i, :])
                outputs.append(q_values)
            return outputs

        # 测试模式：每个节点观测张量为 1D
        elif isinstance(x, list) and x[0].dim() == 1:
            encoded_nodes = []
            for i in range(self.num_nodes):
                xi = x[i].unsqueeze(0)  # 增加 batch 维度 -> [1, obs_dim]
                encoded = self.encoders[i](xi)  # [1, embed_dim]
                encoded_nodes.append(encoded.squeeze(0))  # [embed_dim]
            encoded_stack = torch.stack(encoded_nodes, dim=0)  # [num_nodes, embed_dim]
            x_gc = F.relu(self.conv1(encoded_stack, edge_index))
            # x_gc = F.relu(self.conv2(x_gc, edge_index))


            outputs = []
            for i in range(self.num_nodes):
                q_values = self.decoders[i](x_gc[i])  # [out_dim_i]
                outputs.append(q_values)
            return outputs
        else:
            raise ValueError("GraphDQN 输入格式错误，需为节点列表且每个元素为1D或2D张量。")



# class effect_for_complex(nn.Module):
#     def __init__(self, args):
#         super(effect_for_complex, self).__init__()
#         self.inputs_dim = args.effect_input_dim
#         self.embed_dim = args.hidden_dim
#
#         self.encoder = nn.ModuleList([NodeEncoder(in_dim, self.embed_dim) for in_dim in self.inputs_dim])
#         self.decoder = nn.ModuleList([NodeDecoder(self.embed_dim, 10) for _ in range(len(self.inputs_dim))])
#
#     def forward(self, x_list):
#         # print(x_list,'x_list')
#         encoder_list = []
#         for i, encoder in enumerate(self.encoder):
#             encoder_list.append(encoder(x_list[i]))
#
#         x = torch.stack(encoder_list)
#
#         outputs = []
#         for i, decoder in enumerate(self.decoder):
#             outputs.append(decoder(x[:, i, :]))
#
#         return outputs



class effect_net_(nn.Module):
    def __init__(self, args,input_dim):
        super(effect_net_, self).__init__()
        self.f1 = nn.Linear(input_dim, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim, args.effect_output_dim)

    def forward(self, x):
        x = F.relu(self.f1(x))
        return self.f2(x)
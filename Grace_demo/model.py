import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, projection_hidden_dim, activation=F.relu, base_model=GCNConv, tau=0.5):
        super(Model, self).__init__()
        self.tau = tau
        self.conv1 = base_model(input_dim, output_dim * 2)
        self.conv2 = base_model(output_dim * 2, output_dim)
        self.activation = activation

        self.fc1 = nn.Linear(output_dim, projection_hidden_dim)
        self.fc2 = nn.Linear(projection_hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        return x

    def projection(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x
    
    def loss(self, x1, x2):
        return (self.semi_loss(x1, x2) + self.semi_loss(x2, x1)) / 2 

    def semi_loss(self, x1, x2):
        # 计算任意两个节点之间的余弦相似度
        # eg. [2708, 128] + [2708, 128] -> [2708, 2708]
        between_sim = torch.exp(self.sim(x1, x2) / self.tau)
        refl_sim = torch.exp(self.sim(x1, x1) / self.tau)
        
        # between_sim 的对角是每个节点和自己另一个视图的相似度
        # refl_sim.sum(1) 是每个节点和自己视图的相似度，在 1 维度上求和即把整个矩阵都求和
        # - refl_sim.diag() 是因为，加上 refl_sim.sum(1) 时，对角线上的元素都加上了，我们只希望“同一个视图中”且“和自己不是同一个节点”的样本对作为负例
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())).mean()

    def sim(self, x1, x2):
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        return torch.mm(x1, x2.t())
    
def drop_feature(x, drop_rate):
    drop_mask = torch.empty(x.size(1), dtype=torch.float).uniform_(0, 1) < drop_rate
    drop_mask = drop_mask.to(x.device)
    # 不是很清楚这个 clone 的作用？？
    x = x.clone()
    x[:, drop_mask] = 0
    return x


import torch
from torch_geometric.utils import negative_sampling  # 懒得自己写了
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset

EPS = 1e-15

class InnerProductDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        return x

    def encode(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class GAE(torch.nn.Module):
    def __init__(self, encoder, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)
    
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None): # reconstruction loss
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss
        
    def test(self, z, pos_edge_index, neg_edge_index):
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        pos_y = torch.ones(pos_edge_index.size(1), dtype=torch.float)
        neg_y = torch.zeros(neg_edge_index.size(1), dtype=torch.float)
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0) # get a decimal \in (0, 1)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)
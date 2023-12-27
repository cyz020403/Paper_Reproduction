"""
借鉴 pyg 教程中的写法
https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html?highlight=gae#pytorch-geometric-tutorial-project
"""

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit  # 懒得自己写了

from model import Encoder, InnerProductDecoder, GAE

def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index) # 这里是消息传递的边，都是正边，用 pos_edge_label_index 也是一样的

    # 已经提前从整个图上构造好负边了，如果在 loss 中再构造，可能会导致构造 test 集中的边作为负边，因此这里直接传入更好
    loss = model.recon_loss(z, data.edge_index)
    # loss = model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index)
    loss.backward()
    optimizer.step()
    return loss

def test(data, model):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index) # 消息传递的边，包括 train 的和 test 的部分
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

def main():
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 0.01
    epochs = 100

    # 链接的教程中，这里设置的是 2，如果这里设置为 32 或者 16，可以得到非常高的准确率
    out_channels = 2

    my_seed = 0
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)

    # Load dataset
    dataset = Planetoid("./", "Cora")
    data = dataset[0].to(device)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    # print(data)
    # Data(x=[2708, 1433], edge_index=[2, 10556])

    """
    教程中使用的 train_test_split_edges 已经弃用，推荐用 RandomLinkSplit 代替

    关于 RandomLinkSplit 函数的问题，可以看这个帖子：https://github.com/pyg-team/pytorch_geometric/issues/3668

    注意，add_negative_train_samples 属性默认为 True，因此进行下一步训练的时候需要注意这里添加了负边
    如果 split_labels=False， edge_lable 会标明负边
    如果 split_labels=True，会添加 pos_edge_label 等属性标明正负边

    neg_sampling_ratio 默认为 1，即采样负边的数量与正边总数相同

    注意，当 is_undirected=True 时，认为传入的 edge_index 的后半部分是前半部分的反转，因此处理的过程中会将后半部分的边去掉，
    这样会导致 pos_edge_label 的数量等于原来边总数的一半，neg_edge_label 的数量与之相同
    Cora 数据集显然不满足这个条件，因此这里需要设置为 False

    edge_index 是消息传递用的边，都是正边，其 shape 解释如下（复制于上述帖子）：
    - for training, we exchange messages on all training edges
    - for validation, we exchange messages on all training edges
    - for testing, we exchange messages on all training and validation edges
    """
    transform = RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=False, 
                                split_labels=True, add_negative_train_samples=True)

    train_data, val_data, test_data = transform(data)
    """
    print("Train data: ", train_data)
    print("Test data: ", test_data)
    print("Val data: ", val_data)

    split_labels=False 时：
    Train data:  Data(x=[2708, 1433], edge_index=[2, 8974], edge_label=[17948], edge_label_index=[2, 17948])
    Test data:  Data(x=[2708, 1433], edge_index=[2, 9501], edge_label=[2110], edge_label_index=[2, 2110])
    Val data:  Data(x=[2708, 1433], edge_index=[2, 8974], edge_label=[1054], edge_label_index=[2, 1054])

    split_labels=True 时：
    Train data:  Data(x=[2708, 1433], edge_index=[2, 8974], pos_edge_label=[8974], pos_edge_label_index=[2, 8974], 
                            neg_edge_label=[8974], neg_edge_label_index=[2, 8974])
    Test data:  Data(x=[2708, 1433], edge_index=[2, 9501], pos_edge_label=[1055], pos_edge_label_index=[2, 1055], 
                            neg_edge_label=[1055], neg_edge_label_index=[2, 1055])
    Val data:  Data(x=[2708, 1433], edge_index=[2, 8974], pos_edge_label=[527], pos_edge_label_index=[2, 527], 
                            neg_edge_label=[527], neg_edge_label_index=[2, 527])
    """


    # Model
    encoder = Encoder(in_channels=dataset.num_features, out_channels=out_channels)
    decoder = InnerProductDecoder()
    model = GAE(encoder, decoder)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for epoch in range(1, epochs + 1):
        loss = train(train_data, model, optimizer)
        auc, ap = test(test_data, model)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')

if __name__ == "__main__":
    main()
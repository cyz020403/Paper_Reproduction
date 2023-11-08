import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import my_model

print(torch.__version__)
print(torch.cuda.is_available())

def train(model, optimizer, criterion, x, edge_index, y, idx_train):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[idx_train], y[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, edge_index, y, idx_test):
    model.eval()
    out = model(x, edge_index)
    pred = out.argmax(dim=1)
    pred = pred[idx_test]
    correct = pred.eq(y[idx_test]).sum().item()
    print('correct: ', correct)
    acc = correct / idx_test.sum().item()
    return acc


def main():
    # Hyperparameters
    hidden_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.01
    weight_decay = 5e-4

    # Load dataset
    dataset = Planetoid('./', 'Cora')
    data = dataset[0].to(device)
    # print(data) # Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
    x = data.x
    edge_index = data.edge_index
    y = data.y
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask

    input_dim = dataset.num_features # 1433
    output_dim = dataset.num_classes # 7

    # Define model
    model = my_model.myGCN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    # Train
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, 501):
        loss = train(model, optimizer, criterion, x, edge_index, y, idx_train)
        print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss))
        if epoch % 10 == 0:
            acc = test(model, x, edge_index, y, idx_test)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
            print('Test Accuracy: {:.5f}'.format(acc))
    print('Best Accuracy: {:.5f}, Best Epoch: {:03d}'.format(best_acc, best_epoch))


if __name__ == '__main__':
    main()

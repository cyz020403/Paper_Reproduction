import torch
import torch.nn as nn

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_adj

from model import Model, drop_feature
from eval import Model_Eval

print(torch.__version__)
print(torch.cuda.is_available())

def train(model, optimizer, data, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2):
    model.train()
    optimizer.zero_grad()
    edge_index1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
    edge_index2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]
    x1 = drop_feature(data.x, drop_feature_rate_1)
    x2 = drop_feature(data.x, drop_feature_rate_2)
    x1 = model(x1, edge_index1)
    x2 = model(x2, edge_index2)
    x1 = model.projection(x1)
    x2 = model.projection(x2)
    loss = model.loss(x1, x2)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, num_classes, idx_train, idx_val, idx_test):
    model.eval()
    embenddings = model(data.x, data.edge_index).detach()
    train_embs = embenddings[idx_train]
    val_embs = embenddings[idx_val]
    test_embs = embenddings[idx_test]
    train_lbls = data.y[idx_train]
    val_lbls = data.y[idx_val]
    test_lbls = data.y[idx_test]
    accs = []
    device = embenddings.device

    for _ in range(5):
        model_eval = Model_Eval(model, embenddings.size()[1], num_classes).to(device)
        optimizer_eval = torch.optim.Adam(model_eval.parameters(), lr=0.01, weight_decay=5e-4)

        for _ in range(100):
            model_eval.train()
            optimizer_eval.zero_grad()
            logits = model_eval(train_embs)
            loss = nn.CrossEntropyLoss()(logits, train_lbls)
            loss.backward()
            optimizer_eval.step()

        logits = model_eval(test_embs)
        preds = logits.argmax(dim=1)

        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    accs = torch.tensor(accs)
    print(accs)
    print('Accuracy: {:.4f} ± {:.4f}'.format(torch.mean(accs), torch.std(accs)))
    # print max and min
    print('Max: {:.4f}'.format(torch.max(accs)))
    print('Min: {:.4f}'.format(torch.min(accs)))

    return torch.mean(accs)


def main():
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dim = 128
    projection_hidden_dim = 128
    learning_rate = 0.01
    weight_decay = 5e-4
    num_epochs = 200
    tau = 0.5

    drop_edge_rate_1 = 0.2
    drop_edge_rate_2 = 0.4
    drop_feature_rate_1 = 0.3
    drop_feature_rate_2 = 0.4

    # Load dataset
    dataset = Planetoid('./', 'Cora')
    data = dataset[0].to(device)
    # print(data) # Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask

    # Model definition
    model = Model(data.num_features, output_dim, projection_hidden_dim, tau=tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training
    bestacc = 0
    bestepoch = 0
    for i in range(1, num_epochs + 1):
        loss = train(model, optimizer, data, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2)
        print('Epoch {}, Loss {:.4f}'.format(i, loss))

        # Testing
        if i % 10 == 0:
            acc = test(model, data, dataset.num_classes, idx_train, idx_val, idx_test)
            if acc > bestacc:
                bestacc = acc
                bestepoch = i
                # torch.save(model.state_dict(), 'model.pkl')
                # print('model saved')
            
            print('Best Epoch {}, Best Accuracy {:.4f}'.format(bestepoch, bestacc))
            print('+++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()
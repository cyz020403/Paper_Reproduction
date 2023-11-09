import torch
import torch.nn as nn

class Model_Eval(nn.Module):
    def __init__(self, model, input_dim, output_dim):
        super(Model_Eval, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


    def forward(self, x):
        x = self.fc(x)
        return x
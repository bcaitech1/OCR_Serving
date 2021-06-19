import torch.nn as nn


class FeedForward2D(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(FeedForward2D, self).__init__()

        self.conv_1 = nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1)

        # Depthwise Separable Convolution
        self.dsc = nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1, groups=hidden_dim * 4)

        self.conv_2 = nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.dsc(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv_2(out)
        out = self.dropout(out)

        return out


class FeedForward1D(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(FeedForward1D, self).__init__()

        self.linear_1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear_2 = nn.Linear(hidden_dim * 4, hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear_2(out)
        out = self.dropout(out)

        return out

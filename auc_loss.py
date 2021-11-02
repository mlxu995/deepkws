# multi-class AUC loss
import torch
import torch.nn.functional as F


class AucLoss(torch.nn.Module):

    def __init__(self, gama=0.3, unk_label=0):
        super().__init__()
        self.gama = gama
        self.unk_label = unk_label

    def forward(self, x, y):
        x = torch.sigmoid(x)
        y_onehot = F.one_hot(y, x.shape[1]).bool()
        # in this loss, the output probability corresponding "unk" class is not used
        x_mask = torch.zeros_like(x).bool()
        x_mask[:, self.unk_label] = True
        y_onehot[:, self.unk_label] = False

        positive = torch.masked_select(x, y_onehot)
        negative = torch.max(torch.masked_fill(x, y_onehot|x_mask, 0), dim=-1)[0]

        margain = positive.unsqueeze(-1) - negative.unsqueeze(0) - self.gama
        margain = torch.masked_select(margain, margain < 0)
        loss = torch.sum(torch.pow(margain, 2)) / (positive.shape[0] + 1) / (negative.shape[0] + 1)

        return loss
        
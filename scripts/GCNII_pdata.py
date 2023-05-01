import argparse
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import log
from torch_geometric.nn import GCN2Conv
import torchmetrics
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Citeseer')
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--list_number', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
dataset = Planetoid(path, args.dataset, split='public', transform=transform)
data = dataset[0]
data.adj_t = gcn_norm(data.adj_t)

list_num=args.list_number
if list_num==16:
    list_cir = [2,1,1,1,1,1,1]
if list_num==32:
    list_cir = [4,4,4,3,3,3,3]
if list_num==64:
    list_cir = [8,8,8,8,8,8,8]
if list_num==128:
    list_cir = [18,17,17,17,17,17,17]


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()
        torch.manual_seed(12345)
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1, shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=args.hidden_channels, num_layers=62, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6).to(device)
data = data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.convs.parameters(), weight_decay=0.01),
    dict(params=model.lins.parameters(), weight_decay=5e-4)
], lr=args.lr) 


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.adj_t).argmax(dim=-1)

    precision = torchmetrics.Precision(task="multiclass", average='weighted', num_classes=dataset.num_classes).to(device)
    pre = precision(pred[data.test_mask], data.y[data.test_mask])
    recall = torchmetrics.Recall(task="multiclass", average='weighted', num_classes=dataset.num_classes).to(device)
    rec = recall(pred[data.test_mask], data.y[data.test_mask])
    f1score = torchmetrics.F1Score(task="multiclass", average='weighted', num_classes=dataset.num_classes).to(device)
    f1 = f1score(pred[data.test_mask], data.y[data.test_mask])

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs[0], accs[1], accs[2], pre, rec, f1


best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    loss = train()
    train_acc, val_acc, tmp_test_acc, tmp_test_precision, tmp_test_recall, tmp_test_f1 = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        test_precision = tmp_test_precision
        test_recall = tmp_test_recall
        test_f1 = tmp_test_f1
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
print(best_val_acc, test_acc, test_precision, test_recall, test_f1)
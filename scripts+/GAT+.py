import argparse
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv
import torchmetrics
from utils import GTV


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--list_number', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, split='public', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

list_num=args.list_number
if list_num==16:
    list_cir = [2,1,1,1,1,1,1]
if list_num==32:
    list_cir = [4,4,4,3,3,3,3]
if list_num==64:
    list_cir = [8,8,8,8,8,8,8]
if list_num==128:
    list_cir = [18,17,17,17,17,17,17]

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        torch.manual_seed(12345)
        # Cora and Citeseer
        self.firstlinear = torch.nn.Linear(in_channels, 2**10)
        self.secondlinear = torch.nn.Linear(2**11, 2**10)
        self.thirdlinear = torch.nn.Linear(2**10,2**9)
        self.forthlinear = torch.nn.Linear(2**9,2**8)
        self.fivelinear = torch.nn.Linear(2**8,2**7)
        self.sixlinear = torch.nn.Linear(2**7,2**6)
        self.sevenlinear = torch.nn.Linear(2**6,2**5)
        self.conv1 = GATConv(in_channels, 128, heads, dropout=0.6)
        self.conv1_sub = torch.nn.ModuleList()

     # Pubmed
        # self.firstlinear = torch.nn.Linear(in_channels, 256)
        # self.secondlinear = torch.nn.Linear(256, 128)
        # self.thirdlinear = torch.nn.Linear(128,64)
        # self.forthlinear = torch.nn.Linear(64,32)
        # self.fivelinear = torch.nn.Linear(32,16)
        # self.sixlinear = torch.nn.Linear(16,8)
        # self.sevenlinear = torch.nn.Linear(8,8)

        for _ in range(list_cir[0]):
            # self.conv1_sub.append(GATConv(128 * heads, 128, heads, dropout=0.6)) # Cora
            self.conv1_sub.append(GATConv(128* heads, 128, heads, dropout=0.6)) # Citeseer

        self.conv2 = GATConv(128 * heads, 64, heads, dropout=0.6)
        self.conv2_sub = torch.nn.ModuleList()
        for _ in range(list_cir[1]):
            # self.conv2_sub.append(GATConv(64 * heads, 64, heads, dropout=0.6)) # Cora
            self.conv2_sub.append(GATConv(64 * heads, 64, heads, dropout=0.6)) # Citeseer

        self.conv3 = GATConv(64 * heads, 32, heads, dropout=0.6)
        self.conv3_sub = torch.nn.ModuleList()
        for _ in range(list_cir[2]):
            # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) # Cora
            self.conv3_sub.append(GATConv(32 * heads, 32, heads, dropout=0.6)) # Citeseer

        self.conv4 = GATConv(32 * heads, 16, heads, dropout=0.6)
        self.conv4_sub = torch.nn.ModuleList()
        for _ in range(list_cir[3]):
            # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) # Cora
            self.conv4_sub.append(GATConv(16 * heads, 16, heads, dropout=0.6)) # Citeseer

        self.conv5 = GATConv(16 * heads, 8, heads, dropout=0.6)
        self.conv5_sub = torch.nn.ModuleList()
        for _ in range(list_cir[4]):
            # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) # Cora
            self.conv5_sub.append(GATConv(8 * heads, 8, heads, dropout=0.6)) # Citeseer

        self.conv6 = GATConv(8 * heads, 8, heads, dropout=0.6)
        self.conv6_sub = torch.nn.ModuleList()
        for _ in range(list_cir[5]):
            # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) # Cora
            self.conv6_sub.append(GATConv(8 * heads, 8, heads, dropout=0.6)) # Citeseer

        self.conv7 = GATConv(8 * heads, 4, heads, dropout=0.6)
        self.conv7_sub = torch.nn.ModuleList()
        for _ in range(list_cir[6]):
            # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) # Cora
            self.conv7_sub.append(GATConv(4 * heads, 4, heads, dropout=0.6)) # Citeseer

        # # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv8 = GATConv(4 * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        z = torch.zeros(x.shape[0]-1, (x.shape[-1]))
        z = z.to(x.device) 
        x, z = GTV(x,z) 
        x = F.tanh(self.conv1(x, edge_index))
        for conv in self.conv1_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index)
        z = self.firstlinear(z)
        x, z = GTV(x,z) 
        x = F.tanh(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.tanh(self.conv2(x, edge_index))
        for conv in self.conv2_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index)
        z = self.thirdlinear(z)
        x, z = GTV(x,z) 
        x=F.tanh(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.tanh(self.conv3(x, edge_index))
        for conv in self.conv3_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index)
        z = self.forthlinear(z)
        x, z = GTV(x,z) 
        x=F.tanh(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.tanh(self.conv4(x, edge_index))
        for conv in self.conv4_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index)
        z = self.fivelinear(z)
        # x, z = GTV(x,z) 
        x=F.tanh(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.tanh(self.conv5(x, edge_index))
        for conv in self.conv5_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index)
        z = self.sixlinear(z)
        x, z = GTV(x,z) 
        x=F.tanh(x)    

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.tanh(self.conv6(x, edge_index))
        for conv in self.conv6_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index)
        # z = self.sevenlinear(z)
        x, z = GTV(x,z) 
        x=F.tanh(x)  

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.tanh(self.conv7(x, edge_index))
        for conv in self.conv7_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index)
        z = self.sevenlinear(z)
        x, z = GTV(x,z) 
        x=F.tanh(x)  
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv8(x, edge_index)

        return x


model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes, args.heads).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

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
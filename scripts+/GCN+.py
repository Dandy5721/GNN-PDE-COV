import argparse
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import torchmetrics
from utils import GTV

SEED=2
# np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) 
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--list_number', type=int, default=8)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')
init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
           hidden_channels=args.hidden_channels, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, split='public', transform=T.NormalizeFeatures())
data = dataset[0]

list_num=args.list_number # for different layer
if list_num==16:
    list_cir = [2,1,1,1,1,1,1]
if list_num==32:
    list_cir = [4,4,4,3,3,3,3]
if list_num==64:
    list_cir = [8,8,8,8,8,8,8]
if list_num==128:
    list_cir = [18,17,17,17,17,17,17]

if args.use_gdc:
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
    data = transform(data)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        # Cora and Citeseer
        self.firstlinear = torch.nn.Linear(in_channels, 2**11)
        self.secondlinear = torch.nn.Linear(2**11, 2**10)
        self.thirdlinear = torch.nn.Linear(2**10,2**9)
        self.forthlinear = torch.nn.Linear(2**9,2**8)
        self.fivelinear = torch.nn.Linear(2**8,2**7)
        self.sixlinear = torch.nn.Linear(2**7,2**6)
        self.sevenlinear = torch.nn.Linear(2**6,2**5)
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, 2048, cached=True, normalize=not args.use_gdc)
        # self.conv1_sub = torch.nn.ModuleList()
        # for _ in range(list_cir[0]):
        #     # self.conv1_sub.append(GCNConv(2**10, 2**10, cached=True, normalize=not args.use_gdc)) # Cora
        #     self.conv1_sub.append(GCNConv(2048, 2048, cached=True, normalize=not args.use_gdc)) # Citeseer

        self.conv2 = GCNConv(2048, 1024, cached=True,  normalize=not args.use_gdc)
        # self.conv2_sub = torch.nn.ModuleList()
        # for _ in range(list_cir[1]):
        #     # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) # Cora
        #     self.conv2_sub.append(GCNConv(1024, 1024, cached=True, normalize=not args.use_gdc)) # Citeseer

        self.conv3 = GCNConv(1024, 512, cached=True,  normalize=not args.use_gdc)
        # self.conv3_sub = torch.nn.ModuleList()
        # for _ in range(list_cir[2]):
        #     # self.conv3_sub.append(GCNConv(2**8, 2**8, cached=True, normalize=not args.use_gdc)) # Cora
        #     self.conv3_sub.append(GCNConv(512, 512, cached=True, normalize=not args.use_gdc)) # Citeseer

        self.conv4 = GCNConv(512, 256, cached=True,  normalize=not args.use_gdc)
        # self.conv4_sub = torch.nn.ModuleList()
        # for _ in range(list_cir[3]):
        #     # self.conv4_sub.append(GCNConv(2**7, 2**7, cached=True, normalize=not args.use_gdc)) # Cora
        #     self.conv4_sub.append(GCNConv(256, 256, cached=True, normalize=not args.use_gdc)) # Citeseer

        self.conv5 = GCNConv(256, 128, cached=True,  normalize=not args.use_gdc)
        # self.conv5_sub = torch.nn.ModuleList()
        # for _ in range(list_cir[4]):
        #     # self.conv5_sub.append(GCNConv(2**6, 2**6, cached=True, normalize=not args.use_gdc)) # Cora
        #     self.conv5_sub.append(GCNConv(128, 128, cached=True, normalize=not args.use_gdc)) # Citeseer

        self.conv6 = GCNConv(128, 64, cached=True,  normalize=not args.use_gdc)
        # self.conv6_sub = torch.nn.ModuleList()
        # for _ in range(list_cir[5]):
        #     # self.conv6_sub.append(GCNConv(2**5, 2**5, cached=True, normalize=not args.use_gdc)) # Cora
        #     self.conv6_sub.append(GCNConv(64, 64, cached=True, normalize=not args.use_gdc)) # Citeseer

        self.conv7 = GCNConv(64, 32, cached=True, normalize=not args.use_gdc)
        # self.conv7_sub = torch.nn.ModuleList()
        # for _ in range(list_cir[6]):
        #     self.conv7_sub.append(GCNConv(32, 32, cached=True, normalize=not args.use_gdc))

        self.conv8 = GCNConv(32, out_channels, cached=True, normalize=not args.use_gdc)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        z = torch.zeros(x.shape[0]-1, (x.shape[-1]))
        x, z = GTV(x,z) 
        x = self.conv1(x, edge_index, edge_weight).relu()
        # for conv in self.conv1_sub:
        #     x = F.dropout(x, p=0.5, training=self.training)
        #     x = conv(x, edge_index, edge_weight).relu()
        z = self.firstlinear(z)

        # z = z.to(x.device) 
        # x, z = GTV(x,z)   

        # z = self.firstlinear(z)
        # x, z = GTV(x,z)
        # x = x.tanh()

        x = F.dropout(x, p=0.5, training=self.training)
        # z = self.secondlinear(z)
        # x, z = GTV(x,z)
        x = self.conv2(x, edge_index, edge_weight).relu()
        # for conv in self.conv2_sub:
        #     x = F.dropout(x, p=0.5, training=self.training)
        #     x = conv(x, edge_index, edge_weight).relu()

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index, edge_weight).relu()
        # for conv in self.conv3_sub:
        #     x = F.dropout(x, p=0.5, training=self.training)
        #     x = conv(x, edge_index, edge_weight).relu()

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index, edge_weight).relu()
        # for conv in self.conv4_sub:
        #     x = F.dropout(x, p=0.5, training=self.training)
        #     x = conv(x, edge_index, edge_weight).relu()

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv5(x, edge_index, edge_weight).relu()
        # for conv in self.conv5_sub:
        #     x = F.dropout(x, p=0.5, training=self.training)
        #     x = conv(x, edge_index, edge_weight).relu()

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv6(x, edge_index, edge_weight).relu()
        # for conv in self.conv6_sub:
        #     x = F.dropout(x, p=0.5, training=self.training)
        #     x = conv(x, edge_index, edge_weight).relu()

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv7(x, edge_index, edge_weight).relu()
        # for conv in self.conv7_sub:
        #     x = F.dropout(x, p=0.5, training=self.training)
        #     x = conv(x, edge_index, edge_weight).relu()

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv8(x, edge_index, edge_weight)
        return x


model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes)
print(model)
model, data = model.to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
# optimizer = torch.optim.Adam([
#     dict(params=model.conv1.parameters(), weight_decay=5e-4),
#     dict(params=model.conv2.parameters(), weight_decay=0)
# ], lr=args.lr)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

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
import argparse
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv, DeepGCNLayer, GENConv
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import torchmetrics
from torch.nn import LayerNorm, Linear, ReLU
from utils import GTV
import scipy.io as io

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--number_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=2500)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--use_conda', type=str, default='cuda:1')
parser.add_argument('--GTV_number', type=int, default=1)
parser.add_argument('--internal_number', type=int, default=4)

args = parser.parse_args()

device = torch.device(args.use_conda if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
           hidden_channels=args.hidden_channels, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, split='public', transform=T.NormalizeFeatures())
data = dataset[0]
print(data)


class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        torch.manual_seed(12345)
        self.node_encoder = Linear(dataset.num_features, hidden_channels)
        self.Norm = LayerNorm(hidden_channels * (2**num_layers), elementwise_affine=True)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels* (2**(i-1)), hidden_channels* (2**(i-1)), aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels* (2**(i-1)), elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='dense', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels * (2**num_layers), dataset.num_classes)
        self.lins = Linear(hidden_channels, hidden_channels*2)
    def forward(self, x, edge_index):
        GTV_number = args.GTV_number
        
        internal = (args.number_layer)//(args.internal_number)

        if GTV_number == 1:        
            x = self.node_encoder(x)
            # edge_attr = self.edge_encoder(edge_attr)
            z = torch.zeros(x.shape[0]-1, (x.shape[-1]))
            z = z.to(x.device) 
            x, z = GTV(x,z) 
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers:
                x = layer(x, edge_index)

            x = self.Norm(x)
            x = self.layers[0].act(x)
            # x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)
        
        elif GTV_number == 0: 
            x = self.node_encoder(x)
            # edge_attr = self.edge_encoder(edge_attr)

            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers:
                x = layer(x, edge_index)

            x = self.Norm(x)
            x = self.layers[0].act(x)
            # x = self.layers[0].act(self.layers[0].norm(x))
            xx = F.dropout(x, p=0.1, training=self.training)    


        elif GTV_number == 2:        
            x = self.node_encoder(x)
            # edge_attr = self.edge_encoder(edge_attr)
            z = torch.zeros(x.shape[0]-1, (x.shape[-1]))
            z = z.to(x.device) 
            x, z = GTV(x,z) 
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[0:internal-1]:
                x = layer(x, edge_index)
            z=self.lins(z)
            x, z = GTV(x,z)
            
            for layer in self.layers[internal:]:
                x = layer(x, edge_index)
          

            x = self.Norm(x)
            x = self.layers[0].act(x)
            # x = self.layers[0].act(self.layers[0].norm(x))
            xx = F.dropout(x, p=0.1, training=self.training)         


        elif GTV_number == 3:        
            x = self.node_encoder(x)
            # edge_attr = self.edge_encoder(edge_attr)
            z = torch.zeros(x.shape[0]-1, (x.shape[-1]))
            z = z.to(x.device) 
            x, z = GTV(x,z) 
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[0:internal-1]:
                x = layer(x, edge_index)
            z=self.lins(z)
            x, z = GTV(x,z)

            for layer in self.layers[internal:2*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)

            for layer in self.layers[2*internal:]:
                x = layer(x, edge_index)
          

            x = self.Norm(x)
            x = self.layers[0].act(x)
            # x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)  

        elif GTV_number == 4:        
            x = self.node_encoder(x)
            # edge_attr = self.edge_encoder(edge_attr)
            z = torch.zeros(x.shape[0]-1, (x.shape[-1]))
            z = z.to(x.device) 
            x, z = GTV(x,z) 
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[0:internal-1]:
                x = layer(x, edge_index)
            z=self.lins(z)
            x, z = GTV(x,z)

            for layer in self.layers[internal:2*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)

            for layer in self.layers[2*internal:3*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)            

            for layer in self.layers[3*internal:]:
                x = layer(x, edge_index)
          

            x = self.Norm(x)
            x = self.layers[0].act(x)
            # x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)


        elif GTV_number == 5:        
            x = self.node_encoder(x)
            # edge_attr = self.edge_encoder(edge_attr)
            z = torch.zeros(x.shape[0]-1, (x.shape[-1]))
            z = z.to(x.device) 
            x, z = GTV(x,z) 
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[0:internal-1]:
                x = layer(x, edge_index)
            z=self.lins(z)
            x, z = GTV(x,z)

            for layer in self.layers[internal:2*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)

            for layer in self.layers[2*internal:3*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)            

            for layer in self.layers[3*internal:4*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)  

            for layer in self.layers[4*internal:]:
                x = layer(x, edge_index)
          

            x = self.Norm(x)
            x = self.layers[0].act(x)
            # x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training) 

        elif GTV_number == 6:        
            x = self.node_encoder(x)
            # edge_attr = self.edge_encoder(edge_attr)
            z = torch.zeros(x.shape[0]-1, (x.shape[-1]))
            z = z.to(x.device) 
            x, z = GTV(x,z) 
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[0:internal-1]:
                x = layer(x, edge_index)
            z=self.lins(z)
            x, z = GTV(x,z)

            for layer in self.layers[internal:2*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)

            for layer in self.layers[2*internal:3*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)            

            for layer in self.layers[3*internal:4*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)  

            for layer in self.layers[4*internal:5*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z) 

            for layer in self.layers[5*internal:]:
                x = layer(x, edge_index)

            x = self.Norm(x)
            x = self.layers[0].act(x)
            # x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training) 

        elif GTV_number == 7:        
            x = self.node_encoder(x)
            # edge_attr = self.edge_encoder(edge_attr)
            z = torch.zeros(x.shape[0]-1, (x.shape[-1]))
            z = z.to(x.device) 
            x, z = GTV(x,z) 
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[0:internal-1]:
                x = layer(x, edge_index)
            z=self.lins(z)
            x, z = GTV(x,z)

            for layer in self.layers[internal:2*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)

            for layer in self.layers[2*internal:3*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)            

            for layer in self.layers[3*internal:4*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z)  

            for layer in self.layers[4*internal:5*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z) 

            for layer in self.layers[5*internal:6*internal-1]:
                x = layer(x, edge_index)
            x, z = GTV(x,z) 

            for layer in self.layers[6*internal:]:
                x = layer(x, edge_index)

            x = self.Norm(x)
            x = self.layers[0].act(x)
            # x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)  


        return self.lin(x),xx


device = torch.device(args.use_conda if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=args.hidden_channels, num_layers=args.number_layer-2).to(device)
data = data.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out,out_f = model(data.x, data.edge_index)

    x_train = (data.x).cpu().numpy()
    dataset1=args.dataset
    # print(dataset)
    io.savemat(''+dataset1+'/x_train_DenseGCN.mat', {'data': x_train})

    if args.GTV_number == 0:
        out_fea_ori = out_f.cpu().detach().numpy()
        io.savemat(''+dataset1+'/out_train_ori_DenseGCN.mat', {'data': out_fea_ori})
  
    elif args.GTV_number == 2:
        out_fea_ours = out_f.cpu().detach().numpy()
        io.savemat(''+dataset1+'/out_train_DenseGCN.mat', {'data': out_fea_ours})
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred,out_f = model(data.x, data.edge_index)
    pred = pred.argmax(dim=-1)
    dataset1=args.dataset
    x_train = (data.x).cpu().numpy()
    io.savemat(''+dataset1+'/x_test_DenseGCN.mat', {'data': x_train})

    if args.GTV_number == 0:
        out_fea_ori = out_f.cpu().detach().numpy()
        io.savemat(''+dataset1+'/out_test_ori_DenseGCN.mat', {'data': out_fea_ori})
  
    elif args.GTV_number == 2:
        out_fea_ours = out_f.cpu().detach().numpy()
        io.savemat(''+dataset1+'/out_test_DenseGCN.mat', {'data': out_fea_ours})
        
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
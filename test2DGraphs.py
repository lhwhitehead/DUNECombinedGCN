import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import MNISTSuperpixels
from multiViewGCN import multiViewGCN
import torch.nn.functional as F
import torch_geometric.transforms as T


# We can use a predifined dataset to play around with things.
transform = T.Cartesian(cat=False)
dataset = MNISTSuperpixels('mnistDataset/',True,transform=transform)
dataset = dataset.shuffle()
print('Dataset has ' + str(len(dataset)) + ' entries')

graphsPerBatch = 64

n = len(dataset) // 10
test_dataset = dataset[0:100]
train_dataset = dataset[100:2100]
test_loader = DataLoader(test_dataset, batch_size=graphsPerBatch)
train_loader = DataLoader(train_dataset, batch_size=graphsPerBatch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data0 = dataset[2].to(device)
data1 = dataset[9].to(device)
data2 = dataset[20].to(device)
myGCN = multiViewGCN(train_loader.dataset.num_node_features).to(device)
#optimizer = torch.optim.Adam(myGCN.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = torch.optim.Adam(myGCN.parameters(), lr=0.5, weight_decay=5e-4)

nEpochs = 20

def train():
    myGCN.train()
    loss_all = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = myGCN(data,data,data)

        loss = F.nll_loss(out, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

def test(loader):
    myGCN.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = myGCN(data,data,data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(nEpochs):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, test_acc))

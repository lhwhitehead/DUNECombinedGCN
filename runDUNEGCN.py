import os
import numpy
import torch
from torch_geometric.data import Data, DataLoader
from multiViewGCN import multiViewGCN
from createTorchDataSet import DuneGCNDataset
from createTorchDataSet import DuneGCNProcess
from torch_geometric.nn import DataParallel
import torch.nn.functional as F
import torch_geometric.transforms as T

# Let's see if we can be reproducable!
torch.manual_seed(11)

# Load our dataset
processor = DuneGCNProcess('../graphs/',shuffle=True, energy_cut=20.0)
processor.create_datasets()

datasets = [processor.dataset_view0,processor.dataset_view1,processor.dataset_view2]

# Use 10% for testing
test_frac = 0.1
ntest = int(len(datasets[0])*test_frac)
test_size = ntest
train_size = ntest*9

print('Number of graphs = ',len(datasets[0]))
print('Number of test graphs = ',test_size)
print('Number of train graphs =',train_size)
#print('First tests =',datasets[0][0],datasets[1][0],datasets[2][0])
#print('Last test =',datasets[0][test_size-1],datasets[1][test_size-1],datasets[2][test_size-1])
#print('First train =',datasets[0][test_size],datasets[1][test_size],datasets[2][test_size])
#print('Last train =',datasets[0][train_size+test_size],datasets[1][train_size+test_size],datasets[2][train_size+test_size])

# Set the batch size and divide up the training and test graphs
graphsPerBatch = 64

# Make data loaders for the samples
print('Making data loaders using batch size ',graphsPerBatch)
# Set up all of the test and train data loaders
test_loader  = [DataLoader(datasets[0][0:test_size-1],batch_size=graphsPerBatch),
                DataLoader(datasets[1][0:test_size-1],batch_size=graphsPerBatch),
                DataLoader(datasets[2][0:test_size-1],batch_size=graphsPerBatch)]
train_loader = [DataLoader(datasets[0][test_size:test_size+train_size],batch_size=graphsPerBatch),
                DataLoader(datasets[1][test_size:test_size+train_size],batch_size=graphsPerBatch),
                DataLoader(datasets[2][test_size:test_size+train_size],batch_size=graphsPerBatch)]

#print('Can we use GPU? ',torch.cuda.is_available())
# Select which GPUs we can see
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device.current_device())
myGCN = multiViewGCN(2,4,device)

# Use multiple GPUs if we can
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    myGCN = DataParallel(myGCN)

optimizer = torch.optim.Adam(myGCN.parameters(), lr=0.0005) #, weight_decay=5e-4)

nEpochs = 20

def train():
    myGCN.train()
    loss_all = 0
    loss_func = torch.nn.CrossEntropyLoss()

    for data0, data1, data2 in zip(train_loader[0],train_loader[1],train_loader[2]):
        data0 = data0.to(device)
        data1 = data1.to(device)
        data2 = data2.to(device)
#        print(data0.y,data1.y,data2.y)
        optimizer.zero_grad()
        out = myGCN(data0,data1,data2)
#        print(out,data0.y)
        loss = loss_func(out, data0.y)
        loss.backward()
        loss_all += data0.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len(train_loader[0].dataset)

def test(loader):
    myGCN.eval()
    correct = 0
    nEachFlavour = numpy.zeros([4],numpy.float)#[0.,0.,0.,0.]
    correctEachFlavour = numpy.zeros([4],numpy.float)#[0.,0.,0.,0.]

    for data0, data1, data2 in zip(loader[0],loader[1],loader[2]):
        data0 = data0.to(device)
        data1 = data1.to(device)
        data2 = data2.to(device)
        pred = myGCN(data0,data1,data2)
#        print(pred)
        pred = torch.softmax(pred,1)
#        print(pred)
        pred = pred.max(1)[1]
#        pred = myGCN(data).max(1)[1]
        correct += pred.eq(data0.y).sum().item()
#        print(pred,data0.y)

        # Specific flavour calculations
        for single_correct, single_flavour in zip(pred.eq(data0.y),data0.y):
            nEachFlavour[single_flavour] += 1
            if single_correct == 1:
                correctEachFlavour[single_flavour] += 1

    flavourCorrect = correctEachFlavour / (nEachFlavour)

    return correct / len(loader[0].dataset), flavourCorrect

print('Starting training...')
for epoch in range(nEpochs):
    thisloss = train()
    train_acc,tr_flav_acc = test(train_loader)
    test_acc,te_flav_acc = test(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, thisloss, train_acc, test_acc))
    print('\t Train Acc :: CC numu: {:.5f}, CC nue:  {:.5f}, CC nutau: {:.5f}, NC: {:.5f}'.
            format(tr_flav_acc[0],tr_flav_acc[1],tr_flav_acc[2],tr_flav_acc[3]))
    print('\t Test Acc  :: CC numu: {:.5f}, CC nue:  {:.5f}, CC nutau: {:.5f}, NC: {:.5f}'.
            format(te_flav_acc[0],te_flav_acc[1],te_flav_acc[2],te_flav_acc[3]))


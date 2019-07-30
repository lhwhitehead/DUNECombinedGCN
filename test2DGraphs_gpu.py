import os
import numpy
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import MNISTSuperpixels
from multiViewGCN import multiViewGCN
from createDataSet import duneGraph
import torch.nn.functional as F
import torch_geometric.transforms as T

# Let's see if we can be reproducable!
torch.manual_seed(11)

# Create the DUNE dataset
graphCollection = []
nGraphs = 0
graphLimit = 100000000

runtypes = ['nutau','nue','numu']
#runtypes = ['nutau','nue','numu','anutau','anue','anumu']
graphCount = [0,0,0,0]

for filetype in runtypes: 
    subdirs = os.listdir("graphData/"+filetype)
    for i in range(len(subdirs)):
        allfiles = os.listdir("graphData/"+filetype+"/"+subdirs[i])
        for f in range(len(allfiles)):
    
            tokens = allfiles[f].split('_')
            if tokens[2] != '0.gz':
                continue
            graphLoader = duneGraph("graphData/"+filetype,subdirs[i],tokens[1])
            data0, data1, data2 = graphLoader.getGraphs()
            keep_graph = True
            if data0.y == 3:
                if numpy.random.uniform() < 0.50:
                    keep_graph = False

            if keep_graph == True:
                nGraphs += 1
                graphCollection.append([data0,data1,data2])
                graphCount[data0.y] += 1
    
            if nGraphs >= graphLimit:
                break
        if nGraphs >= graphLimit:
            break
    if nGraphs >= graphLimit:
        break

# Since we will likely load numu / nue / nutau sequentially then we should shuffle them
numpy.random.shuffle(graphCollection)

# We need to transpose our matrix so that each row contains one view
graphCollection = list(map(list, zip(*graphCollection)))
print("Number of graphs = " + str(len(graphCollection[0])))
print(graphCount)

# Set the batch size and divide up the training and test graphs
graphsPerBatch = 64
n = len(graphCollection[0]) // 10
test_size  = n 
train_size = n*9

# Set up all of the test and train data loaders
test_loader  = [DataLoader(graphCollection[0][0:test_size-1],batch_size=graphsPerBatch),
                DataLoader(graphCollection[1][0:test_size-1],batch_size=graphsPerBatch),
                DataLoader(graphCollection[2][0:test_size-1],batch_size=graphsPerBatch)]
train_loader = [DataLoader(graphCollection[0][test_size:test_size+train_size],batch_size=graphsPerBatch),
                DataLoader(graphCollection[1][test_size:test_size+train_size],batch_size=graphsPerBatch),
                DataLoader(graphCollection[2][test_size:test_size+train_size],batch_size=graphsPerBatch)]
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
myGCN = multiViewGCN(2,4,device)#.to(device)
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


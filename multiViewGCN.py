import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

#
# Graph 0 ---> Convolutions ---> |
#                                |
# Graph 1 ---> Convolutions ---> | ---> MLP ---> [p0,p1,p2,p3,p4,p5,p6,p7,p8,p9]
#                                |
# Graph 2 ---> Convolutions ---> |
#

class multiViewGCN(torch.nn.Module):
    def __init__(self,nFeatures,nCategories):
        super(multiViewGCN,self).__init__()

        # Convolutions and pooling layers for each input branch
        self.conv0s = [GraphConv(nFeatures,128),GraphConv(nFeatures,128),GraphConv(nFeatures,128)]
        self.pool0s = [TopKPooling(128, ratio=0.8),TopKPooling(128, ratio=0.8),TopKPooling(128, ratio=0.8)]
        self.conv1s = [GraphConv(128,128),GraphConv(128,128),GraphConv(128,128)]
        self.pool1s = [TopKPooling(128, ratio=0.8),TopKPooling(128, ratio=0.8),TopKPooling(128, ratio=0.8)]
        self.conv2s = [GraphConv(128,128),GraphConv(128,128),GraphConv(128,128)]
        self.pool2s = [TopKPooling(128, ratio=0.8),TopKPooling(128, ratio=0.8),TopKPooling(128, ratio=0.8)]

        # MLP takes merged outputs from the three branches
        self.mlp0 = torch.nn.Linear(3*256,2*256)
        self.mlp1 = torch.nn.Linear(2*256,128)
        self.mlp2 = torch.nn.Linear(128,nCategories)


    def forward(self,data0,data1,data2):

        # Get the information for the three graphs in arrays
        xs = [data0.x,data1.x,data2.x]
        edges = [data0.edge_index,data1.edge_index,data2.edge_index]
        batches = [data0.batch,data1.batch,data2.batch]

        # Perform separate convolutions for the three graphs. These three branches each replicate the network described
        # in https://arxiv.org/abs/1810.02244 with the exception of the MLP (we merge our three branches first)
        for branch in range(3):
            xs[branch] = F.relu(self.conv0s[branch](xs[branch],edges[branch]))
            xs[branch], edges[branch], _, batches[branch], _ = self.pool0s[branch](xs[branch], edges[branch], None, batches[branch])
            x1 = torch.cat([gmp(xs[branch], batches[branch]), gap(xs[branch], batches[branch])], dim=1)

            xs[branch] = F.relu(self.conv1s[branch](xs[branch],edges[branch]))
            xs[branch], edges[branch], _, batches[branch], _ = self.pool1s[branch](xs[branch], edges[branch], None, batches[branch])
            x2 = torch.cat([gmp(xs[branch], batches[branch]), gap(xs[branch], batches[branch])], dim=1)

            xs[branch] = F.relu(self.conv2s[branch](xs[branch],edges[branch]))
            xs[branch], edges[branch], _, batches[branch], _ = self.pool2s[branch](xs[branch], edges[branch], None, batches[branch])
            x3 = torch.cat([gmp(xs[branch], batches[branch]), gap(xs[branch], batches[branch])], dim=1)

            # We effectively have some sort of residual connections from each of the graph convolutions
            xs[branch] = x1 + x2 + x3
            
        # Concatenate the outputs from the three graph branches
        x = torch.cat((xs[0],xs[1],xs[2]),1)

        # Run through three dense layers that form the MLP
        x = F.relu(self.mlp0(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.mlp1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp2(x)

#        return F.log_softmax(x,dim=-1)
#        return F.softmax(x,dim=-1)
#        print("Raw     ",x)
#        print("Simple  ",x/1000.)
#        print("Sigmoid ",torch.sigmoid(x))
#        print("Tanh    ",torch.tanh(x))
#        print("Softmax ",torch.softmax(x,1))
#        print("Relu    ",torch.relu(x))
        return x/1000.

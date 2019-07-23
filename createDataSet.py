import torch
import zlib
import numpy as np
from torch_geometric.data import Data
from torch import Tensor
from torch_geometric.transforms import KNNGraph

# Class to wrap up the truth information
class truthInfo:
    def __init__(self):
        super(truthInfo,self).__init__()

        self.interaction  = 0
        self.pdg          = 0
        self.protons      = 0
        self.pions        = 0
        self.pizeros      = 0
        self.neutrons     = 0

    def getFlavour(self):
        if self.interaction < 4: 
            return 0 # numu
        elif self.interaction < 8:
            return 1 # nue
        elif self.interaction < 12:
            return 2 # nutau
        else: 
            return 3 # NC

class duneGraph:
    def __init__(self,path,subdir,event):
        super(duneGraph,self).__init__()

        self.dataFeatures = [0,0,0]
        self.dataPositions = [0,0,0]
        self.truthInfo = truthInfo()

        self.viewNodes = [0,0,0]
        self.nodeFeatures = 0
        self.nodeCoords = 0
        
        # Get the number of nodes and features from the info file
        infofiles = [open(path+"/"+subdir+"/event_"+event+"_0.info","r"),   
                     open(path+"/"+subdir+"/event_"+event+"_1.info","r"),   
                     open(path+"/"+subdir+"/event_"+event+"_2.info","r")]
        
        # Extract the truth information and graph sizes
        self.readInfoFiles(infofiles)

        # Load the data files for this event
        datafiles = [open(path+"/"+subdir+"/event_"+event+"_0.gz","rb"),    
                     open(path+"/"+subdir+"/event_"+event+"_1.gz","rb"),    
                     open(path+"/"+subdir+"/event_"+event+"_2.gz","rb")]

        # Unpack the data files and fill the feature and position tensors
        for d in range(3):
            elements = np.fromstring(zlib.decompress(datafiles[d].read()), dtype=np.float32, sep='')
            elements = elements.reshape((self.viewNodes[d],self.nodeFeatures+self.nodeCoords))
            split = np.hsplit(elements,self.nodeCoords)
            pos = split[0]
            features = split[1]
            for i in range(len(features)):
                if features[i][0] > 1000:
                    features[i][0] = 1000.
                features[i][0] = features[i][0] / 1000.
                if features[i][1] > 10:
                    features[i][1] = 24.
                features[i][1] = features[i][1] / 24.
                #print(features[i][0],features[i][1])
            self.dataFeatures[d] = torch.from_numpy(features)
            self.dataPositions[d] = torch.from_numpy(pos)
            datafiles[d].close();

        y = torch.zeros([1], dtype=torch.long)
        y[0] = self.getFlavour()

        # Now we make the actual graph data objects with the same prediction tensor
        self.data0 = Data(x=self.dataFeatures[0],pos=self.dataPositions[0],y=y)
        self.data1 = Data(x=self.dataFeatures[1],pos=self.dataPositions[1],y=y)
        self.data2 = Data(x=self.dataFeatures[2],pos=self.dataPositions[2],y=y)

        # Use the KNNGraph to calculate our edges - request the default six nearest neighbours
        edgeFinder = KNNGraph(k=6)
        self.data0 = edgeFinder(self.data0)
        self.data1 = edgeFinder(self.data1)
        self.data2 = edgeFinder(self.data2)

    # Function to extract the truth and graph size information from the .info file
    def readInfoFiles(self,infofiles):
        counter = 0
        for infofile in infofiles:
#            print("Reading file " + infofile.name)
            lines = infofile.readlines()
            infofile.close()
            # Only need these lines from one of the three files
            if counter == 0:
                self.truthInfo.interaction = int(lines[0])
                self.truthInfo.pdg = int(lines[7])
                self.truthInfo.protons = int(lines[8])
                self.truthInfo.pions = int(lines[9])
                self.truthInfo.pizeros = int(lines[10])
                self.truthInfo.neutrons = int(lines[11])    

                self.nodeCoords = int(lines[15])
                self.nodeFeatures = int(lines[16])
            
            self.viewNodes[counter] = int(lines[14])

            counter = counter + 1

    # Get the neutrino flavour (reduced version of the interactiopn type)
    def getFlavour(self):
        return self.truthInfo.getFlavour()

    # Return the torch_geometric.data objects for the three views
    def getGraphs(self):
        return self.data0, self.data1, self.data2

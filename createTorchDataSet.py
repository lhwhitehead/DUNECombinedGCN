import os
import os.path as osp
import numpy
import torch
import random
from glob import glob
from torch_geometric.data import Dataset
from createDataSet import duneGraph

# Class to hold together the three graphs per event
class DUNEEventGraphs:
  def __init__(self,data0,data1,data2):
    self.data0 = data0
    self.data1 = data1
    self.data2 = data2
  
  def get_graphs(self):
    return data0,data1,data2

# Define the dataset for each view individually... make sure
# we load the graphs in the same order for each event
class DuneGCNProcess:
  def __init__(self,root,shuffle=False,energy_cut=100.0):
    self.base_dir = root
    self.processed_dir = self.base_dir+'processed'
    self.dataset_view0 = []
    self.dataset_view1 = []
    self.dataset_view2 = []
    self.neighbours=10
    self.ncFrac = 0.45
    self.nevents = 0
    self.subdir_limit = 1000
    self.shuffle=shuffle
    self.energy_cut = energy_cut
  def create_datasets(self):
    print('Processing raw data')

    # We have six types of files to consider
    runtypes = ['nutau','nue','numu','anutau','anue','anumu']

    for filetype in runtypes:
      subdirs = os.listdir("../graphs/"+filetype)
      subdir_num=0
      for i in range(len(subdirs)):

        if subdir_num == self.subdir_limit:
          break
#        print('Reading directory ',filetype+'/'+subdirs[i])
        subdir_num = subdir_num + 1
        allfiles = os.listdir("../graphs/"+filetype+"/"+subdirs[i])
        for f in range(len(allfiles)):

          if os.path.isfile(osp.join(self.processed_dir,'data_view0_{}.pt'.format(self.nevents))) == True:
            continue

          tokens = allfiles[f].split('_')
          if tokens[2] != '0.gz':
            continue
          graphLoader = duneGraph("../graphs/"+filetype,subdirs[i],tokens[1],self.neighbours)

          data0, data1, data2 = graphLoader.getGraphs()
          # We have too many NC events in general, so remove some
          keep_graph = True
          if data0.y == 3:
            if numpy.random.uniform() < self.ncFrac:
              keep_graph = False

          # If the energy is too high then skip it
          if graphLoader.getTrueEnergy > self.energy_cut:
            keep_graph = False

          # Write the three graphs to disk
          if keep_graph == True:
            torch.save(data0,osp.join(self.processed_dir,'data_view0_{}.pt'.format(self.nevents)))
            torch.save(data1,osp.join(self.processed_dir,'data_view1_{}.pt'.format(self.nevents)))
            torch.save(data2,osp.join(self.processed_dir,'data_view2_{}.pt'.format(self.nevents)))
            self.nevents = self.nevents + 1

    # If we have made all the files before, we need to count the number
    if self.nevents == 0:
      self.nevents = int(len(os.listdir(self.processed_dir))/3)

    shuffle_array=None
    if self.shuffle == True:
      shuffle_array = list(range(0,self.nevents))
      random.shuffle(shuffle_array)

    self.dataset_view0 = DuneGCNDataset(self.base_dir,self.processed_dir,view=0,ngraphs=self.nevents,shuffle=self.shuffle,shuffle_array=shuffle_array)
    self.dataset_view1 = DuneGCNDataset(self.base_dir,self.processed_dir,view=1,ngraphs=self.nevents,shuffle=self.shuffle,shuffle_array=shuffle_array)
    self.dataset_view2 = DuneGCNDataset(self.base_dir,self.processed_dir,view=2,ngraphs=self.nevents,shuffle=self.shuffle,shuffle_array=shuffle_array)

# Class that actually stores the dataset for torch-geometric to play with
class DuneGCNDataset(Dataset):
  def __init__(self,base_dir,process_dir,view=0,ngraphs=0,transform=None,pre_transform=None,shuffle=False,shuffle_array=None):
    super(DuneGCNDataset, self).__init__(base_dir,transform,pre_transform)

    self.base_dir=base_dir
    self.processed_dir=process_dir
    self.view = view
    self.ngraphs = ngraphs
    self.shuffle = shuffle
    self.shuffle_array = shuffle_array
    print('Constructing DUNEGCNDataset for view ',self.view)

  @property
  def raw_file_names(self):
   return [] 

  @property
  def processed_file_names(self):
    return glob(self.processed_dir+'*.pt')

  def __len__(self):
    return self.ngraphs
  
  def _download(self):
    pass

  # We process the files earlier as we need three datasets 
  def _process(self):
    pass

  # Load a single event - in our case this is actually three graph objects
  def get(self,idx):
    # Not sure why I have to do this, but maybe it works?
#    idx = idx.stop
    # For some reason idx can be an int or a slice... let's try to do both
    basename='data_view'+str(self.view)
    if type(idx) is int:
      newidx = idx
      if self.shuffle == True:
        newidx = self.shuffle_array[idx]
#      print('Original index = '+str(idx)+' and new idex = ',newidx)
      filename = basename+'_'+str(newidx)+'.pt'
      data = torch.load(osp.join(self.processed_dir,filename))
      return data
    else:
#      print('Getting range of files from '+str(idx.start)+' to '+str(idx.stop))
      data = []
      for i in range(idx.start,idx.stop):
         newidx = i
         if self.shuffle == True:
           newidx = self.shuffle_array[i]
#         print('Original index = '+str(i)+' and new idex = ',newidx)
         filename = basename+'_'+str(newidx)+'.pt'
         if i == idx.start:
           data = [torch.load(osp.join(self.processed_dir,filename))]
         else:
           data.append(torch.load(osp.join(self.processed_dir,filename)))
#      print('Collated data has size ',len(data))
      return data 


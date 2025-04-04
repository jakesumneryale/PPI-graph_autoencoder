##pt_Dataset.py
##Created: 3/03/25

from os import listdir
from os.path import isfile, join
import h5py
import numpy as np
import torch
from torch_geometric.data import Dataset, Data


'''
Creates graph_dataset to load into pytorch readable dataset format
Generates torch_geometric.data.Data from graph created using create_graph_format.py
'''

class graph_dataset(Dataset):
    
    def __init__(self,hdf5_file, decoy_name,node_features, edge_features, target=None, root='./'):
        """
        Args:
            hdf5_file (string/pathlib.Path): path to the .hdf5 file
            decoy_name (string): name of decoy to load
            node_features (list): list of desired node features
            edge_features (list): list of desired edge features
            target (string): name of target score (must match name in .hdf5 file)
            root (string/pathlib.Path): location to save output .pt files


        """
        
        self.hdf5_file = hdf5_file
        self.decoy_name=decoy_name
        self.node_features = node_features
        self.edge_features=edge_features
        self.target=target
        self.root=root

    def get(self):
        data=self.load_graph(self.hdf5_file,self.decoy_name)
        return data



    def load_graph(self,hdf5_file,decoy):
        """
        Loads graph based on the .hdf5 file and decoy name

        Returns torch_geometric.data.Data object
        """

        with h5py.File(hdf5_file,'r') as fh:

            try:
                graph=fh[decoy]
            except:
                print(f'No graph matching decoy named {decoy}')
                return
            
            try:
                # Loads node features from set given in graph_dataset initialization
                # Sets them to the node_attributes (x) of Data object
                num_nodes=len(graph['node_reference'])
                node_feature_array=np.zeros((num_nodes,0))

                for feature in self.node_features:
                    feature_vals=graph['node_features'][feature][()]
                    node_feature_array=np.column_stack((node_feature_array,feature_vals))
                
                node_attributes=torch.tensor(node_feature_array, dtype=torch.float)

            except:
                print(f'Error retrieving node features from Decoy {decoy}')


            
            try:
                # Loads edges and edge features from set given in graph_dataset initialization
                # Sets them to the edge_index and edge_attributes of Data object
               
                edge_array=graph['edge_features']['contacts'][()]
                edge_index=self.reindex_edges(edge_array)

                edge_attributes=self.reindex_edge_features(self,graph,edge_index)


            except:
                print(f'Decoy {decoy} missing edge features')
                return
            
            
            if self.target != None:
            ## If given, loads associated target score of graph and sets it to  y of Data object
                try:
                    target_score_val=graph['target_scores'][self.target][()]
                    target_score=torch.tensor(target_score_val,dtype=torch.float)
                
                except: 
                    print(f'Target score {self.target} not found for decoy {decoy}')
                
        data=Data(x=node_attributes, edge_index=edge_index, edge_attr=edge_attributes, y=target_score)

        return data

            
    
    def reindex_edges(edge_array):
        ## reindexes edges to match required format:
        ## Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        ## num_edges should include both directions of the edge

        reverse_edges=np.flip(edge_array,axis=1)
        all_edges=np.vstack((edge_array,reverse_edges))
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()


        return edge_index
        
    def reindex_edge_features(self,graph,edge_index):
        ## Same as reindexing edge features, reindexes to match required format:
        ## data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        

        edge_feature_array=np.zeros((edge_index.shape[1],0))
        for feature in self.edge_features:

            feature_vals=graph['edge_features'][feature][()]
            reindexed_features=np.vstack((feature_vals,feature_vals))
            edge_feature_array=np.column_stack((edge_feature_array,reindexed_features))

        edge_attributes=torch.tensor(edge_feature_array,dtype=torch.float)

        return edge_attributes

def split_data(hdf5_folder, file_indicator='.hdf5',train_percent=.8,shuffle=False,setseed=None):
    """
    Splits number of .hdf5 files into train and test sets by given percentage
    If shuffle=True, shuffles the files before splitting
    If an integer number is given by setseed, files will be shuffled in the same way for each run


    Splitting the data by file name vs number of graphs in each hdf5 ends up with an average error in the 
    percentage of only +/-.6% with maximums of ~+/- 1.7% (so rather than a .8 split, it may be .806-.817)
    This seems like small enough error to proceed with this method and not take graph counts per file into account
    """


    all_graph_files=sorted([f for f in listdir(hdf5_folder) if isfile(join(hdf5_folder, f)) and file_indicator in f])
    if shuffle:
        if setseed!=None:
            np.random.seed(seed=setseed)
            np.random.shuffle(all_graph_files)
    
        else:
            np.random.shuffle(all_graph_files)

    train,validate = np.split(all_graph_files, [int(len(all_graph_files)*train_percent)])

    return train, validate

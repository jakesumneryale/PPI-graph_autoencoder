##create_graph_format.py
##Created: 2/03/25

from pathlib import Path
import numpy as np
import h5py
import argparse
import pandas as pd
import re
from os import listdir
from os.path import isfile, join
import create_protein_graph_structure as jk
import bounded_voronoi_contacts_radical as voro


sdt = h5py.string_dtype(encoding='utf-8')

def initialize_graphs(pdb_id,pdb_dir,save_dir = "./", file_indicator = "_H_0001.pdb",probe_size=1.4):
    '''
    Initializes .hdf5 formatted graphs from a directory of pdbs for GNN, including a list of the nodes and contacts, 
    along with basic features regarding node, chain, and edge types

    Input:
    pdb_id: Name of protein
    pdb_dir: Directory of pdb files
    save_dir: Directory to save graphs
    file_indicator: Unique identifier of pdb files
    probe_size: Size of probe to be used in generating bounded voronoi tesselation

    Output:
    .hdf5 file containing formatted graph for each pdb file provided in the directory
    '''


    graph_fh=Path(save_dir) / Path(f'{pdb_id}.hdf5')
    fh=h5py.File(str(graph_fh),'w')

    all_decoys = sorted([f for f in listdir(pdb_dir) if isfile(join(pdb_dir, f)) and file_indicator in f])

    for decoy in all_decoys:
        decoy_name = decoy.split(file_indicator)[0]

        try:
            decoy_group = fh.create_group(decoy_name)

            ## Calculating basic information about PPI 

            protein_df=jk.get_protein_information(decoy,pdb_dir)
            bounded_voro_tessellation=voro.get_bounded_voro(protein_df,box_margin=1,dispersion=4.5,probe_size=probe_size)
            all_contacts=voro.get_all_contacts_aa(protein_df=protein_df,voronoi_tessellation=bounded_voro_tessellation)
            neighbor_adj_mat_aa=jk.get_voronoi_neighbors_aa(protein_df, bounded_voro_tessellation)


            ## Initializing node and edge features
            node_df=initialize_node_feats(decoy_group,protein_df,neighbor_adj_mat_aa)
            initialize_edge_feats(decoy_group,node_df,all_contacts)

        except:
            decoy_group=fh[decoy_name]
            print(f"{decoy_name} already initialized")



def initialize_node_feats(decoy_group,protein_df,adj_mat):
    '''
    Creates node feature groups and datasets within a given .hdf5 group
    Adds reference for basic node information (amino acid index, chain onehot encoding, amino acid type)
    Adds datasets of onehot encoding for amino acid type, chain id, interface participation
    '''

    node_feature_group=decoy_group.create_group('node_features')

    node_df=protein_df[['aa_id','chain_id','aa_name']].drop_duplicates().reset_index(drop=True)
    node_list=np.array(node_df,dtype='S')

    chain1_len=len(node_df[node_df['chain_id']==1])


    aa_onehot=np.zeros((0,20))
    node_interface_onehot=np.zeros((len(node_df),1))

    for ind in node_df.index:
        aa_onehot_temp=jk.feature_one_hot_amino_acid_type(ind, node_df['aa_name']).T
        aa_onehot=np.vstack((aa_onehot,aa_onehot_temp))

        node_interface_onehot[ind]=jk.feature_interface_one_hot(adj_mat,ind,chain1_len)
        

    decoy_group.create_dataset('node_reference',data=node_list,dtype=sdt)
    node_feature_group.create_dataset('aa_type',data=aa_onehot)
    node_feature_group.create_dataset('chain',data=node_df['chain_id'].values.reshape((len(node_df),1))-1)
    node_feature_group.create_dataset('interface_nodes',node_interface_onehot)

    return node_df

def initialize_edge_feats(decoy_group,node_df,all_contacts):
    '''
    Creates edge feature groups and datasets within a given .hdf5 group
    Adds reference for basic contact information (list of amino acids interacting with their specific identifiers)
    Adds dataset for indices of amino acid interactions, onehot encoding for interface edges
    '''

    edge_feature_group=decoy_group.create_group('edge_features')

    all_contacts['chain_id1']=all_contacts['chain1'].values-1
    all_contacts['chain_id2']=all_contacts['chain2'].values-1
    contact_list1=np.array(all_contacts[['aa_id1','chain_id1','restype1']].values,dtype='S')
    contact_list2=np.array(all_contacts[['aa_id2','chain_id2','restype2']].values,dtype='S')

    contact_list=[[contact_list1[i],contact_list2[i]] for i in range(len(all_contacts))]
    contact_ids=all_contacts[['aa_id1','aa_id2']].values.astype(int)


    contact_interface_onehot=np.zeros((len(all_contacts),1))
    for ind in all_contacts.index:
        if all_contacts.loc[ind]['chain1']==all_contacts.loc[ind]['chain2']:
            contact_interface_onehot[ind]=0 
        else:
            contact_interface_onehot[ind]=1

    decoy_group.create_dataset('edge_reference',data=contact_list,dtype=sdt)
    edge_feature_group.create_dataset('contacts',data=contact_ids)
    edge_feature_group.create_dataset('interface_edges',data=contact_interface_onehot)




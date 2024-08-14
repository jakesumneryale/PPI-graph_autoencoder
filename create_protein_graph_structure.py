import numpy as np
import pandas as pd
import Bio
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
import os
from os import listdir
from os.path import isfile, join, isdir
from sklearn import metrics
from numpy.linalg import norm
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.spatial import distance_matrix
import pickle
import re


##########################################
####### GRAPH GENERATION FUNCTIONS #######
##########################################

def get_protein_coords(pdb_name, pdb_dir):
    '''
    Gets the coordinates of the c-alpha and 
    heavy atoms  - in separate lists for simplicity -
    for each amino acids in the protein entered.
    
    Each chain will be separated into it's own list as well. 
    '''
    
    os.chdir(pdb_dir)
    
    ## Init the pdb parser
    
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    ## Get the target structure 
    
    heterodimer = pdb_parser.get_structure(pdb_name.split(".")[0], pdb_name)
    het_model = heterodimer[0]
    
    dimer_ca = []
    dimer_ha = []
    
    first_chain_bool = True
    
    for chain in het_model:
        temp_chain_list_ca = []
        temp_chain_list_ha = []
        count = 0
        for residue in chain:
            temp_res_list = []
            count +=1
            for atom in residue:
                if "CA" in atom.get_name():
                    temp_chain_list_ca.append(atom.get_coord())
                    temp_res_list.append(atom.get_coord())
                    
                elif "H" not in atom.get_name()[0] and atom.get_name()[0] not in ["1", "2", "3", "4"]:
                    temp_res_list.append(atom.get_coord())
                    
            temp_chain_list_ha.append(np.array(temp_res_list))
        
        print("Chain count:", count)
        dimer_ca.append(np.array(temp_chain_list_ca))
        dimer_ha.append(temp_chain_list_ha)
    
    return dimer_ca, dimer_ha
        
## Create the adjacency matrix from the heavy atom coordinates

def get_distances_between_heavy_atom_lists(atoms1, atoms2, cutoff_distance=4.5):
    '''
    Returns 1 if the cutoff distance between the two
    amino acids in atoms1 and atoms2 is below 4.5 for any
    of the atoms in the array
    '''
    
    aa_dist_mat = distance_matrix(atoms1, atoms2)
        
    norm_mat = np.where(aa_dist_mat <= cutoff_distance, 1, 0)
    
    return np.max(norm_mat)

def create_adjacency_matrix(dimer_ha, cutoff_distance=4.5):
    '''
    Creates the adjacency matrix from the heavy atoms of each dimer.
    First the intra-chain distances are calculated and then the 
    inter-chain distances.
    
    The edges are created between residues that are within the cutoff_distance
    from one another, which is set to 4.5Å by default. Covalent bonds also count as edges.
    '''
    
    ## Define the chains
    
    chain1 = dimer_ha[0]
    chain2 = dimer_ha[1]
    
    chain1_len = len(chain1)
    chain2_len = len(chain2)
    tot_len = chain1_len + chain2_len
    
    ## Create the adjacency matrix structure
    
    final_adj_mat = np.zeros((tot_len, tot_len))
    
    ## Loop through each amino acid and add it to the adjacency matrix
    
    ## Loop through chain 1 first and do intra chain distances
    
    for i in range(chain1_len):
        res_i = chain1[i]
        for j in range(i,chain1_len):
            res_j = chain1[j]
            
            edge_val = get_distances_between_heavy_atom_lists(res_i, res_j, cutoff_distance = cutoff_distance)
            
            final_adj_mat[i,j] = edge_val
            final_adj_mat[j,i] = edge_val
        
    ## Loop through chain 2 now and do intra chain distances
    
    for i in range(chain2_len):
        res_i = chain2[i]
        for j in range(i,chain2_len):
            res_j = chain2[j]
            
            new_i = i + chain1_len
            new_j = j + chain1_len
            
            edge_val = get_distances_between_heavy_atom_lists(res_i, res_j, cutoff_distance = cutoff_distance)
            
            final_adj_mat[new_i,new_j] = edge_val
            final_adj_mat[new_j,new_i] = edge_val
        
    
    ## Now do the inter chain distances
    
    for i in range(chain1_len):
        res_i = chain1[i]
        for j in range(chain2_len):
            res_j = chain2[j]
            
            new_ind = j + chain1_len
            
            edge_val = get_distances_between_heavy_atom_lists(res_i, res_j, cutoff_distance = cutoff_distance)
            
            final_adj_mat[i,new_ind] = edge_val
            final_adj_mat[new_ind, i] = edge_val
            
    return final_adj_mat
        
    


def create_pdb_from_ca_coords(dimer_ca, adj_mat, output_file_start, edge_type='all'):
    '''
    Creates a pdb file that has single spheres
    over the C-alpha coordinates for each amino acid.
    It also connects the amino acids according to the CONECT record
    '''
    
    ## Make the edge type text lowercase
    
    edge_type = edge_type.lower()
    
    edge_type_accept_list = ["all", "interface", "intrachain"]
    
    if edge_type not in edge_type_accept_list:
        print("ERROR: PLEASE ENTER IN AN ACCEPTABLE EDGE TYPE: ALL, INTERFACE, INTRACHAIN")
        return 1
    
    output_file = f"{output_file_start}_{edge_type}.pdb"
    
    ## Begin the loop to write out the nodes
    
    total_i = 0
    
    with open(output_file, 'w') as pdb_file:
        
        ## Write the coordinates for the C-alpha files - which are the node locations!
        pdb_file.write("HEADER    GENERATED GRAPH FOR GNN PROJECT\n")
        for cnum, chain in enumerate(dimer_ca):
            for i, (x, y, z) in enumerate(chain, start=1):
                residue_number = i + total_i
                atom_number = i + total_i
                if cnum == 0:
                    chain_name = "A"
                elif cnum == 1:
                    chain_name = "B"
                pdb_file.write(
                    f"ATOM  {atom_number:5d}  CA  ALA {chain_name}{residue_number:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            pdb_file.write("TER\n")
            total_i += i
            
        ## Specify edges using the adjacency matrix and CONECT
        
        chain1_len = len(dimer_ca[0])
        chain2_len = len(dimer_ca[1])
        total_len = chain1_len + chain2_len
        
        ## The full graph
        
        if edge_type == "all":
        
            for i in range(total_len):
                for j in range(i, total_len):
                    if adj_mat[i,j] == 1 and i != j:
                        pdb_file.write(f"CONECT{i+1:5d}{j+1:5d}\n")

        ## The interface edges only
        
        elif edge_type == "interface":
                    
            for i in range(chain1_len):
                for j in range(chain1_len, total_len):
                    if adj_mat[i,j] == 1 and i != j:
                        pdb_file.write(f"CONECT{i+1:5d}{j+1:5d}\n")
                        
        ## Intrachain edges only (no interface edges)
                        
        elif edge_type == "intrachain":
            
            ## Chain 1
            for i in range(chain1_len):
                for j in range(i,chain1_len):
                    if adj_mat[i,j] == 1 and i != j:
                        pdb_file.write(f"CONECT{i+1:5d}{j+1:5d}\n")
                      
            ## Chain 2
            for i in range(chain1_len,total_len):
                for j in range(i,total_len):
                    if adj_mat[i,j] == 1 and i != j:
                        pdb_file.write(f"CONECT{i+1:5d}{j+1:5d}\n")
            
        
    return 0

#################################################
####### NODE FEATURE GENERATION FUNCTIONS #######
#################################################

def feature_interface_one_hot(adj_mat, node_ind, chain1_len):
	'''
	Simple helper function that determines
	whether the node is an interface node or not.
	An interface node is one that has an edge between
	itself and the node of another chain.

	Check to see if the slice of the adjacency matrix for
	the node index has any edges in the overlap region with the 
	other chain.
	'''

	## Check to see if the node_ind is part of chain 1 or chain 2

	interface_slice = []
	if node_ind < chain1_len:
		## Part of chain 1
		interface_slice = adj_mat[node_ind, chain1_len:]

	elif node_ind <= chain1_len:
		## Part of chain 2
		interface_slice = adj_mat[node_ind, :chain1_len]

	return np.max(interface_slice)


def feature_interface_node_degree(adj_mat, node_ind, chain1_len):
	'''
	Calculates the node degree of nodes that have edges
	that go between chains
	'''

	pass


def feature_intrachain_node_degree(adj_mat, node_ind, chain1_len):
	'''
	Calculates the node degree of nodes that have edges
	that only exist with nodes of the same chain
	'''

	pass



    
def generate_node_features(adj_mat, aa_list):
	'''
	Generates the following node features

	- one-hot encoding for whether each node is an interface atom or not
	- The number of intra-chain contacts (node degree with nodes from the same chain)
	- The number of interface contacts (node degree with nodes from different chains)
	- The chain ID one-hot encoding
	- One-hot amino acid type
	- Amino acid hydrophobicity (normalized between 0 and 1; from Jack Logan and the observed rSASA of residues in proteins)
	- Amino acid charge (according to charged groups at pH=7)
	- Amino acid rSASA value
	- Amino acid normalized volume (normalized so that the values are between 0 and 1; min and max values determined by other analyses)
	- Amino acid normalized surface area (normalized so that the values are between 0 and 1; min and max values are determine by other analyses)
	'''

	pass

def main():
	'''
	Runs the code
	'''


	## Load in the data

	pdb_dir = "/Users/jakesumner/Desktop/PPI Project/supersampling/pdb_targets_ethan"

	pdb_name = "2grn_complex_H.pdb"


	## Create the coordinates and the adjacency matrix

	test_ca, test_ha = get_protein_coords(pdb_name, pdb_dir)

	test_adj_mat = create_adjacency_matrix(test_ha)


	## Save the created graph structure as PDB files

	os.chdir("/Users/jakesumner/Desktop/Graph_neural_network/protein_graph_structures")

	create_pdb_from_ca_coords(test_ca, test_adj_mat, "2grn_graph",edge_type = "all")
	create_pdb_from_ca_coords(test_ca, test_adj_mat, "2grn_graph",edge_type = "intrachain")
	create_pdb_from_ca_coords(test_ca, test_adj_mat, "2grn_graph",edge_type = "interface")











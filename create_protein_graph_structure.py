import numpy as np
import pandas as pd
import Bio
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
import os
from os import listdir
from os.path import isfile, join, isdir
from numpy.linalg import norm
from scipy.spatial import distance_matrix
import pickle
import argparse
from glob import glob 
import h5py


## Initialize Parser

parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", help="The name of the directory containing the PDB files to be read in")
parser.add_argument("--output_file", "-o", help="The name of the output file to save the data to")
parser.add_argument("--output_dir", "-od", default = ".", help="The directory where the output file will be saved")
parser.add_argument("--file_indicator", "-fi", default = "", help="The fragment of the file that is used by glob to identify the files in the directory")

##########################################
####### GRAPH GENERATION FUNCTIONS #######
##########################################


aa_three_to_one = {
	'ALA': 'A',  # Alanine
	'ARG': 'R',  # Arginine
	'ASN': 'N',  # Asparagine
	'ASP': 'D',  # Aspartic acid
	'CYS': 'C',  # Cysteine
	'GLU': 'E',  # Glutamic acid
	'GLN': 'Q',  # Glutamine
	'GLY': 'G',  # Glycine
	'HIS': 'H',  # Histidine
	'ILE': 'I',  # Isoleucine
	'LEU': 'L',  # Leucine
	'LYS': 'K',  # Lysine
	'MET': 'M',  # Methionine
	'PHE': 'F',  # Phenylalanine
	'PRO': 'P',  # Proline
	'SER': 'S',  # Serine
	'THR': 'T',  # Threonine
	'TRP': 'W',  # Tryptophan
	'TYR': 'Y',  # Tyrosine
	'VAL': 'V',  # Valine
}

def get_protein_coords(pdb_name, pdb_dir):
	'''
	Gets the coordinates of the c-alpha and 
	heavy atoms  - in separate lists for simplicity -
	for each amino acids in the protein entered.
	
	Each chain will be separated into it's own list as well. 
	'''

	## Load in global dictionary

	global aa_three_to_one
	
	os.chdir(pdb_dir)
	
	## Init the pdb parser
	
	pdb_parser = Bio.PDB.PDBParser(QUIET = True)

	## Get the target structure 
	
	heterodimer = pdb_parser.get_structure(pdb_name.split(".pd")[0], pdb_name)
	het_model = heterodimer[0]
	
	dimer_ca = []
	dimer_ha = []
	aa_list = []
	
	first_chain_bool = True
	
	for chain in het_model:
		temp_chain_list_ca = []
		temp_chain_list_ha = []
		count = 0
		for residue in chain:
			temp_res_list = []
			count +=1
			aa_list.append(aa_three_to_one[residue.get_resname()])
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
	
	return dimer_ca, dimer_ha, aa_list
		
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

	elif node_ind >= chain1_len:
		## Part of chain 2
		interface_slice = adj_mat[node_ind, :chain1_len]

	return np.max(interface_slice)


def feature_interface_node_degree(adj_mat, node_ind, chain1_len):
	'''
	Calculates the node degree of nodes that have edges
	that go between chains
	'''

	interface_slice = []
	if node_ind < chain1_len:
		## Part of chain 1
		interface_slice = adj_mat[node_ind, chain1_len:]

	elif node_ind >= chain1_len:
		## Part of chain 2
		interface_slice = adj_mat[node_ind, :chain1_len]

	return np.sum(interface_slice)
	


def feature_intrachain_node_degree(adj_mat, node_ind, chain1_len):
	'''
	Calculates the node degree of nodes that have edges
	that only exist with nodes of the same chain
	'''

	intrachain_slice = []
	if node_ind < chain1_len:
		## Part of chain 1
		intrachain_slice = adj_mat[node_ind, :chain1_len]

	elif node_ind >= chain1_len:
		## Part of chain 2
		intrachain_slice = adj_mat[node_ind, chain1_len:]

	return np.sum(intrachain_slice)


def feature_chain_id(node_ind, chain1_len, total_len):
	'''
	Returns the chain ID as a one-hot encoding.
	0 for the first chain and 1 for the second chain.
	If value == -1 when returned, then there has been an error with the
	node index not fitting within the total_len.
	'''

	value = -1

	if node_ind < chain1_len:
		value = 0

	elif node_ind >= chain1_len and node_ind < total_len:
		value = 1

	else:
		print("NODE INDEX DOES NOT FIT IN THE RANGE OF THE PROTEIN!")

	return value


#### Global amino acid reference list

amino_acid_reference_list = [
	'A',  # Alanine
	'R',  # Arginine
	'N',  # Asparagine
	'D',  # Aspartic acid
	'C',  # Cysteine
	'E',  # Glutamic acid
	'Q',  # Glutamine
	'G',  # Glycine
	'H',  # Histidine
	'I',  # Isoleucine
	'L',  # Leucine
	'K',  # Lysine
	'M',  # Methionine
	'F',  # Phenylalanine
	'P',  # Proline
	'S',  # Serine
	'T',  # Threonine
	'W',  # Tryptophan
	'Y',  # Tyrosine
	'V',  # Valine
]


def feature_one_hot_amino_acid_type(node_ind, aa_list):
	'''
	Returns the one-hot encoding for the amino acid type
	based on the internal amino acid list, which is
	loaded in as a global variable. 
	'''

	global amino_acid_reference_list

	one_hot_arr = np.zeros((20,1))

	node_aa_id = aa_list[node_ind]

	temp_ind = amino_acid_reference_list.index(node_aa_id)

	temp_one_hot = np.zeros((20,1))

	temp_one_hot[temp_ind] = 1
	
	return temp_one_hot


amino_acid_charges = {
	'A': 0.0,   # Alanine
	'R': 1.0,   # Arginine
	'N': 0.0,   # Asparagine
	'D': -1.0,  # Aspartic acid
	'C': 0.0,   # Cysteine
	'E': -1.0,  # Glutamic acid
	'Q': 0.0,   # Glutamine
	'G': 0.0,   # Glycine
	'H': 0.1,   # Histidine (slight positive charge at pH=7)
	'I': 0.0,   # Isoleucine
	'L': 0.0,   # Leucine
	'K': 1.0,   # Lysine
	'M': 0.0,   # Methionine
	'F': 0.0,   # Phenylalanine
	'P': 0.0,   # Proline
	'S': 0.0,   # Serine
	'T': 0.0,   # Threonine
	'W': 0.0,   # Tryptophan
	'Y': 0.0,   # Tyrosine
	'V': 0.0,   # Valine
}


def feature_amino_acid_charge(node_ind, aa_list):
	'''
	Returns the charge of the amino acid at pH=7

	There are only true charges for Lysine, arginine, glutamic acid,
	aspartic acid, and histidine
	'''

	global amino_acid_charges

	node_aa_id = aa_list[node_ind]

	return amino_acid_charges[node_aa_id]


#### Global amino acid hydrophobicity list based on average rSASA in globular protein structures
## Generated by Jack Logan while he was in the O'Hern group in July, 2024
## Protein reference dataset Dunbrack 1.8

hydrophobicity_amino_acids = {
"G":0.2500902917869621,			# Glycine
"S":0.2708353481758574,			# Serine
"R":0.34002633193307735,		# Arginine
"H":0.2586518659901074,			# Histidine
"C":0.08633063696493479,		# Cysteine
"I":0.10845863173451231,		# Isoleucine
"L":0.12230885833109705,		# Leucine
"A":0.19539296425943847,		# Alanine
"F":0.11501191219925694,		# Phenylalanine
"E":0.39495274243422235,		# Glutamic Acid
"P":0.31578744678312554,		# Proline
"D":0.3518794983211057,			# Aspartic Acid
"K":0.4386713687017281,			# Lysine
"V":0.1250604785552286,			# Valine
"T":0.25474539494546705,		# Threonine
"W":0.13668272842571094,		# Tryptophan
"Q":0.3409964277131179,			# Glutamine
"Y":0.1625991043603646,			# Tyrosine
"N":0.32528534862840985,		# Asparagine
"M":0.15040324614124448			# Methonine
}

def feature_amino_acid_hydrophobicity(node_ind, aa_list):
	'''
	Returns the amino acid hydrophobicity from the

	'''

	global hydrophobicity_amino_acids

	node_aa_id = aa_list[node_ind]

	return hydrophobicity_amino_acids[node_aa_id]


	
def generate_node_features(adj_mat, aa_list, chain1_len, chain2_len):
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

	## The total number of features in the array
	NUM_FEATURES = 6 				# excluded the amino acid one-hot ID (20 long vector) because I don't like it. It's bad.

	total_len = chain1_len + chain2_len

	## Create the feature array

	feature_arr = np.zeros((total_len, NUM_FEATURES))


	## Loop through the features to build up the array


	for node_ind in range(total_len):

		## Interface Amino Acid

		interface_aa_1h = feature_interface_one_hot(adj_mat, node_ind, chain1_len)

		## Interface edges node degree - how many edges connected to the node are "interface" edges?

		interface_degree = feature_interface_node_degree(adj_mat, node_ind, chain1_len)

		## Intra chain node degree - how many edges are with amino acids that belong to the same chain? 

		intrachain_degree = feature_intrachain_node_degree(adj_mat, node_ind, chain1_len)

		## Which chain does the amino acid belong to?

		chain_id_1h = feature_chain_id(node_ind, chain1_len, total_len)

		## Amino acid charge

		aa_charge = feature_amino_acid_charge(node_ind, aa_list)

		## Amino acid hydrophobicity

		aa_hydrophobicity = feature_amino_acid_hydrophobicity(node_ind, aa_list)

		## Add the data to the matrix

		feature_arr[node_ind,0] = interface_aa_1h
		feature_arr[node_ind,1] = interface_degree
		feature_arr[node_ind,2] = intrachain_degree
		feature_arr[node_ind,3] = chain_id_1h
		feature_arr[node_ind,4] = aa_charge
		feature_arr[node_ind,5] = aa_hydrophobicity

	return feature_arr


def save_file_to_hdf5_group(hdf_file, pdb_filename, adj_mat, graph_node_features):
	'''
	Takes the pdb_filename and creates a new group in the hdf_file with that name.
	Then the code adds subfolders for the "A" (adjacency matrix) and "X" (node features), 
	to the initial group so that we can have all the data for each target in the same spot. 
	'''

	group_name = pdb_filename.split(".pd")[0]

	new_group = hdf_file.create_group(group_name)

	new_group.create_dataset("A", data=adj_mat)
	new_group.create_dataset("X", data=graph_node_features)

	return


def main():
	'''
	Runs the code
	'''

	## Parse arguments

	args = parser.parse_args()
	pdb_dir = args.directory
	save_file_name = args.output_file
	file_indicator = args.file_indicator
	save_file_dir = args.output_dir

	## Load in the data

	os.chdir(pdb_dir)

	pdb_files = glob(f"*{file_indicator}*")


	## Open a new HDF5 file to save the data to

	os.chdir(save_file_dir)

	hdf_file = h5py.File(save_file_name, "w")


	## loop through all the PDBs in the directory

	for pdb_filename in pdb_files:


		## Create the coordinates and the adjacency matrix

		dimer_ca, dimer_ha, aa_list = get_protein_coords(pdb_filename, pdb_dir)

		dimer_adj_mat = create_adjacency_matrix(dimer_ha)

		## Create the node features for the graph

		chain1_len = len(dimer_ca[0])
		chain2_len = len(dimer_ca[1])

		graph_node_features = generate_node_features(dimer_adj_mat, aa_list, chain1_len, chain2_len)

		## Save data as an HDF5 dataset file

		save_file_to_hdf5_group(hdf_file, pdb_filename, dimer_adj_mat, graph_node_features)


	## Close the file

	hdf_file.close()



	## Save the created graph structure as PDB files

	# os.chdir("/Users/jakesumner/Desktop/Graph_neural_network/protein_graph_structures")

	# create_pdb_from_ca_coords(dimer_ca, dimer_adj_mat, "2grn_graph",edge_type = "all")
	# create_pdb_from_ca_coords(dimer_ca, dimer_adj_mat, "2grn_graph",edge_type = "intrachain")
	# create_pdb_from_ca_coords(dimer_ca, dimer_adj_mat, "2grn_graph",edge_type = "interface")


if __name__ == '__main__':
	main()










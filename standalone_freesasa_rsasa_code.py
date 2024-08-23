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
import matplotlib.pyplot as plt
import argparse
from glob import glob
import freesasa

## Initialize Parser

parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", help="The name of the directory containing the PDB file(s) to be read in")
parser.add_argument("--input_file", "-f", help="The name of the input file (if only running rSASA on one protein)")
parser.add_argument("--output_dir", "-od", default = ".", help="The directory where the output file will be saved")
parser.add_argument("--file_indicator", "-fi", default = ".pdb", help="The fragment of the file that is used by glob to identify the files in the directory (if running on multiple files). (Default is '.pdb')")
parser.add_argument("--probe_radius", "-pr", default = 1.4, type = float, help="The radius of the probe used by the Lee-Richards algorithm to calculate SASA (uses 1.4 Ang by default)")
parser.add_argument("--nslices", "-ns", default = 100, type = int, help="The number of slices used to determine the surface area for the Lee-Richards algorithm (uses 100 slices by default)")
parser.add_argument("--all_dir", "-a", default = False, type = bool, help="Whether the code is running on an entire directory or just one file (False by default). If true, then file_indicator is used to identify files in the directory to run the code on")
parser.add_argument("--output_csv", "-oc", default = True, type = bool, help="Whether to output the rSASA data as a CSV file (True) or XLSX file (False)")

## Dictionary of atomic radii according to Jennifer's radii (specified in Gaines et al. 2018)

atomic_radius_dict = {
	'A': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'H': 1.0, 'HA': 1.1, 'HB1': 1.1, 'HB2': 1.1, 'HB3': 1.1}, 
	'R': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'CD': 1.5, 'NE': 1.3, 'CZ': 1.3, 'NH1': 1.3, 'NH2': 1.3, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HD2': 1.1, 'HD3': 1.1, 'HG2': 1.1, 'HG3': 1.1, 'HE': 1.0, 'HH11': 1.0, 'HH12': 1.0, 'HH21': 1.0, 'HH22': 1.0}, 
	'D': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.3, 'OD1': 1.4, 'OD2': 1.4, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1}, 
	'N': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.3, 'OD1': 1.4, 'ND2': 1.3, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HD21': 1.0, 'HD22': 1.0}, 
	'C': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'SG': 1.75, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HG': 1.0}, 
	'E': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'CD': 1.3, 'OE1': 1.4, 'OE2': 1.4, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HG2': 1.1, 'HG3': 1.1, 'HE2': 1.0}, 
	'Q': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'CD': 1.3, 'OE1': 1.4, 'NE2': 1.3, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HE21': 1.0, 'HE22': 1.0, 'HG2': 1.1, 'HG3': 1.1}, 
	'G': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'H': 1.0, 'HA1': 1.1, 'HA2': 1.1, 'HA3': 1.1}, 
	'H': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'ND1': 1.3, 'CD2': 1.5, 'CE1': 1.5, 'NE2': 1.3, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HD1': 1.1, 'HD2': 1.1, 'HE1': 1.1, 'HE2': 1.1}, 
	'I': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG1': 1.5, 'CG2': 1.5, 'CD1': 1.5, 'H': 1.0, 'HA': 1.1, 'HB': 1.1, 'HD11': 1.1, 'HD12': 1.1, 'HD13': 1.1, 'HG12': 1.1, 'HG13': 1.1, 'HG21': 1.1, 'HG22': 1.1, 'HG23': 1.1}, 
	'L': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'CD1': 1.5, 'CD2': 1.5, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HG': 1.1, 'HD11': 1.1, 'HD12': 1.1, 'HD13': 1.1, 'HD21': 1.1, 'HD22': 1.1, 'HD23': 1.1}, 
	'K': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'CD': 1.5, 'CE': 1.5, 'NZ': 1.3, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HD2': 1.1, 'HD3': 1.1, 'HE2': 1.1, 'HE3': 1.1, 'HG2': 1.1, 'HG3': 1.1, 'HZ1': 1.0, 'HZ2': 1.0, 'HZ3': 1.0}, 
	'M': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'SD': 1.75, 'CE': 1.5, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HG2': 1.1, 'HG3': 1.1, 'HG1': 1.1, 'HE1': 1.1, 'HE2': 1.1, 'HE3': 1.1}, 
	'F': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'CD1': 1.5, 'CD2': 1.5, 'CE1': 1.5, 'CE2': 1.5, 'CZ': 1.5, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HD1': 1.1, 'HD2': 1.1, 'HE1': 1.1, 'HE2': 1.1, 'HZ': 1.1}, 
	'P': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'CD': 1.5, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HD3': 1.1, 'HD2': 1.1, 'HG2': 1.1, 'HG3': 1.1}, 
	'S': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'OG': 1.4, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HG': 1.0}, 
	'T': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'OG1': 1.4, 'CG2': 1.5, 'H': 1.0, 'HA': 1.1, 'HB': 1.1, 'HG1': 1.0, 'HG21': 1.1, 'HG22': 1.1, 'HG23': 1.1}, 
	'W': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'CD1': 1.5, 'CD2': 1.5, 'NE1': 1.3, 'CE2': 1.5, 'CE3': 1.5, 'CZ2': 1.5, 'CZ3': 1.5, 'CH2': 1.5, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HD1': 1.1, 'HD2': 1.1, 'HE1': 1.0, 'HE2': 1.1, 'HE3': 1.1, 'HZ2': 1.1, 'HZ3': 1.1, 'HH2': 1.1}, 
	'Y': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'CD1': 1.5, 'CD2': 1.5, 'CE1': 1.5, 'CE2': 1.5, 'CZ': 1.5, 'OH': 1.4, 'H': 1.0, 'HA': 1.1, 'HB2': 1.1, 'HB3': 1.1, 'HD1': 1.1, 'HD2': 1.1, 'HE1': 1.1, 'HE2': 1.1, 'HH': 1.0}, 
	'V': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG1': 1.5, 'CG2': 1.5, 'H': 1.0, 'HA': 1.1, 'HB': 1.1, 'HG11': 1.1, 'HG12': 1.1, 'HG13': 1.1, 'HG21': 1.1, 'HG22': 1.1, 'HG23': 1.1}, 
	'Z': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'AD1': 1.5, 'AD2': 1.5}, 
	'X': {'N': 1.3, 'CA': 1.5, 'C': 1.3, 'O': 1.4, 'CB': 1.5, 'CG': 1.5, 'CD': 1.5, 'AE1': 1.5, 'AE2': 1.5}
}

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
    'GLX': 'X',  # Unsure Glutamine/glutamic acid
    'ASX': 'Z',  # Unsure Asparagine/aspartic acid
    'ACE': 'J',  # Just in case
}

def get_protein_information(pdb_name, pdb_dir):
    '''
    Gets all the information for a protein and
    stores it in a pandas dataframe. This includes
    all atoms, their names, radii, coordinates, and 
    which chain they are in 
    '''

    ## Load in global dictionaries
    
    global aa_three_to_one
    global atomic_radius_dict
    
    os.chdir(pdb_dir)
    
    ## Init the pdb parser
    
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    ## Get the target structure 
    
    heterodimer = pdb_parser.get_structure(pdb_name.split(".")[0], pdb_name)
    het_model = heterodimer[0]
    
    amino_acid_id = []
    amino_acid_name = []
    chain_id = []
    atom_name = []
    atom_radius = []
    atomic_x = []
    atomic_y = []
    atomic_z = []
    ca_bool = []
    ha_bool = []
    hyd_bool = []
    
    first_chain_bool = True
    chain_count = 1
    residue_count = 0
    
    for chain in het_model:
    
        for residue in chain:
            
            for atom in residue:
                
                ## Save the chain ID
                chain_id.append(chain_count)
        
                ## Save the amino acid name
                curr_amino_acid = aa_three_to_one[residue.get_resname()]
                amino_acid_name.append(curr_amino_acid)

                ## Save the amino acid ID
                amino_acid_id.append(residue_count)
                
                ## Save the atom name
                curr_atom_name = atom.get_name()
                atom_name.append(curr_atom_name)
                
                ## Save the amino acid coords
                temp_coords = list(atom.get_coord())
                atomic_x.append(temp_coords[0])
                atomic_y.append(temp_coords[1])
                atomic_z.append(temp_coords[2])
                
                ## Specify C-alpha
                if "CA" == curr_atom_name:
                    ca_bool.append(1)
                    ha_bool.append(1)
                    hyd_bool.append(0)
                    
                    atom_radius.append(atomic_radius_dict[curr_amino_acid][curr_atom_name])
                    
                ## Specify Heavy Atom
                elif "H" not in curr_atom_name[0] and curr_atom_name[0] not in ["1", "2", "3", "4"]:
                    ca_bool.append(0)
                    ha_bool.append(1)
                    hyd_bool.append(0)
                    
                    if curr_atom_name == "OXT":
                        atom_radius.append(1.4)
                    else:
                        ## Save the atomic radius for the heavy atoms
                        atom_radius.append(atomic_radius_dict[curr_amino_acid][curr_atom_name])
                    
                ## Specify Hydrogen
                else:
                    ca_bool.append(0)
                    ha_bool.append(0)
                    hyd_bool.append(1)
                    
                    if curr_atom_name in atomic_radius_dict[curr_amino_acid]:
                        atom_radius.append(atomic_radius_dict[curr_amino_acid][curr_atom_name])
                        
                    else:
                        atom_radius.append(1.1) ## slightly larger hydrogen as a failsafe
                        
            residue_count += 1
                            
        
        ## Increment the chain ID
        chain_count += 1
        
        
    ## Create dataframe from information
    
    protein_dict = {
        "chain_id" : chain_id,
        "aa_id" : amino_acid_id,
        "aa_name" : amino_acid_name,
        "atom_name" : atom_name,
        "atom_radius" : atom_radius,
        "x_coord" : atomic_x, 
        "y_coord" : atomic_y, 
        "z_coord" : atomic_z, 
        "ca_bool" : ca_bool,
        "ha_bool" : ha_bool, 
        "hyd_bool" : hyd_bool 
    }
    
    protein_df = pd.DataFrame(protein_dict)
    
    return protein_df


def calculate_sasa_protein(protein_df, new_params):
    '''
    Calculates the SASA of each residue in the context
    of the protein using FreeSASA. The atomic
    radii in the protein_df are used, which are consistent
    with those defined by Jennifer Gaines and used 
    in NACCESS
    '''

    ## Get coordinates for protein

    protein_coords = np.zeros((len(protein_df), 3))
    protein_coords[:,0] = protein_df["x_coord"]
    protein_coords[:,1] = protein_df["y_coord"]
    protein_coords[:,2] = protein_df["z_coord"]
    flat_coords = protein_coords.flatten()

    ## Get Radii

    protein_radii = protein_df["atom_radius"]

    ## Calculate rSASA per atom

    flat_coords = list(flat_coords)
    protein_radii = list(protein_radii)

    sasa_result = freesasa.calcCoord(flat_coords, protein_radii, parameters = new_params)

    
    sasa_atomic = np.zeros((sasa_result.nAtoms(), 1))
    for i in range(sasa_result.nAtoms()):
        sasa_atomic[i] = sasa_result.atomArea(i)

    protein_df["atom_sasa"] = sasa_atomic

    print(sasa_result.totalArea())

    return 0

def calculate_residue_sasa_protein(protein_df):
    '''
    Returns a numpy array with the total SASA for
    each residue according the the residue id
    for each residue in the protein_df
    '''

    residue_sasa = np.zeros((np.max(protein_df["aa_id"]+1),))

    for i in range(len(residue_sasa)):
        temp_df = protein_df[protein_df["aa_id"] == i]
        res_sum = np.sum(temp_df["atom_sasa"])
        residue_sasa[i] = res_sum

    return residue_sasa


def calculate_residue_sasa_solvent(protein_df, new_params):
    '''
    Calculates the SASA of the residue
    when fully solvated. Technically it calculates
    the SASA of the dipeptide, which includes the 
    nitrogen-1 and carbon +1 that is connected to 
    the amino acid via peptide bonding. The only 
    exception would be the first and last residues,
    which only include one of the atoms instead 
    of both.
    '''

    residue_sasa = np.zeros((np.max(protein_df["aa_id"]+1),))
    
    after_list = ["N", "CA", "H"]
    before_list = ["C", "O", "CA"]

    for i in range(len(residue_sasa)):
        temp_df = protein_df[protein_df["aa_id"] == i]

        ## Get temp residue coords

        temp_res_coords = np.zeros((len(temp_df), 3))
        temp_res_coords[:, 0] = temp_df["x_coord"]
        temp_res_coords[:, 1] = temp_df["y_coord"]
        temp_res_coords[:, 2] = temp_df["z_coord"]
        flat_coords = temp_res_coords.flatten()

        ## Get temp residue radii

        temp_radii = temp_df["atom_radius"]

        ## Cast them as lists

        flat_coords = list(flat_coords)
        temp_radii = list(temp_radii)

        ## Add the extra atoms to the lists
        
        extra_atom_count = 0

        if i == 0:
            ## First residue, no C from before
            after_res = protein_df[protein_df["aa_id"] == i+1]
            try:
                for k in range(3):
                    after_res_atom = after_res[after_res["atom_name"] == after_list[k]].iloc[0]
                    
                    flat_coords += [after_res_atom["x_coord"], after_res_atom["y_coord"], after_res_atom["z_coord"]]
                    temp_radii.append(after_res_atom["atom_radius"])
                    extra_atom_count +=1 ## atom count incremented if this step was successful
            except:
                ## A catch for issues with proline
                pass


        elif i == len(residue_sasa)-1:
            ## Last residue, no N from next
            before_res = protein_df[protein_df["aa_id"] == i-1]

            try:
                for k in range(3):
                    before_res_atom = before_res[before_res["atom_name"] == before_list[k]].iloc[0]
                    
                    flat_coords += [before_res_atom["x_coord"], before_res_atom["y_coord"], before_res_atom["z_coord"]]
                    temp_radii.append(before_res_atom["atom_radius"])
                    extra_atom_count +=1 ## atom count incremented if this step was successful
            except:
                ## A catch for issues with proline
                pass

        else:
            ## Normal residue, add both C before and N after
            before_res = protein_df[protein_df["aa_id"] == i-1]
            after_res = protein_df[protein_df["aa_id"] == i+1]
            
            ## Before
            try:
                for k in range(3):
                    before_res_atom = before_res[before_res["atom_name"] == before_list[k]].iloc[0]
                    
                    flat_coords += [before_res_atom["x_coord"], before_res_atom["y_coord"], before_res_atom["z_coord"]]
                    temp_radii.append(before_res_atom["atom_radius"])
                    extra_atom_count +=1 ## atom count incremented if this step was successful
            except:
                ## A catch for issues with proline
                pass
            
            ## After
            try:
                for k in range(3):
                    after_res_atom = after_res[after_res["atom_name"] == after_list[k]].iloc[0]
                    
                    flat_coords += [after_res_atom["x_coord"], after_res_atom["y_coord"], after_res_atom["z_coord"]]
                    temp_radii.append(after_res_atom["atom_radius"])
                    extra_atom_count +=1 ## atom count incremented if this step was successful
            except:
                ## A catch for issues with proline
                pass



        ## Calculate the SASA

        sasa_result = freesasa.calcCoord(flat_coords, temp_radii, parameters = new_params)
        
        ## Only add up the surface areas of the one amino acid
        
        temp_area = sasa_result.totalArea()
        tot_len = sasa_result.nAtoms()
        
        if i == 0 or i == len(residue_sasa)-1:
            for j in range(1,extra_atom_count + 1):
                temp_area -= sasa_result.atomArea(tot_len-j) ## subtract the 3 off of the end
        else:
            for j in range(1,extra_atom_count + 1):
                temp_area -= sasa_result.atomArea(tot_len-j) ## subtract the 6 off of the end

        ## Save the data

        residue_sasa[i] = temp_area

    return residue_sasa

def get_aa_list(protein_df):
    '''
    Simply gets a list of the amino acids
    in the protein, in order, not separated by
    chain at all
    '''
    
    aa_list = []
    
    
    num_max = np.max(protein_df["aa_id"])+1
    
    for i in range(num_max):
        
        aa_list.append(protein_df[protein_df["aa_id"] == i]["aa_name"].iloc[0])
        
    return aa_list


#############################
####### MAIN FUNCTION #######
#############################

def main():
	'''
	The logic to run the code
	'''

	## Parse the args

	args = parser.parse_args()

	input_dir = args.directory
	all_dir_bool = args.all_dir
	file_indicator = args.file_indicator

	os.chdir(input_dir)

	if all_dir_bool:
		pdb_files = glob(f"*{file_indicator}*")

	else:
		pdb_file = args.input_file

	probe_radius = args.probe_radius
	nslices = args.nslices

	output_csv_bool = args.output_csv
	output_dir = args.output_dir

	## Specific the parameters 

	new_params = freesasa.Parameters.defaultParameters

	new_params["n-slices"] = nslices
	new_params["probe-radius"] = probe_radius

	new_params = freesasa.Parameters(new_params)

	## Running on a single file

	if not all_dir_bool:

		## Get the protein information dataframe 

		protein_df = get_protein_information(pdb_file, input_dir)

		## Get the SASA for the protein

		calculate_sasa_protein(protein_df, new_params)

		res_sasa = calculate_residue_sasa_protein(protein_df)

		## Get the SASA for the solvated dipeptides

		sol_sasa = calculate_residue_sasa_solvent(protein_df, new_params)

		## Get the rSASA

		rsasa_vals = res_sasa/sol_sasa

		## Create dataframe and organize easily

		rsasa_df = pd.DataFrame()

		rsasa_df["residue_name"] = get_aa_list(protein_df)

		rsasa_df["SASA_protein"] = res_sasa 

		rsasa_df["SASA_dipep"] = sol_sasa

		rsasa_df["rSASA"] = rsasa_vals

		## Save the data to a file

		os.chdir(output_dir)

		if output_csv_bool:

			save_filename = pdb_file.split(".pd")[0] + "_sasa_data.csv"

			rsasa_df.to_csv(save_filename)

		elif not output_csv_bool:

			save_filename = pdb_file.split(".pd")[0] + "_sasa_data.xlsx"

			rsasa_df.to_excel(save_filename)


	## Running on multiple files in a directory

	elif all_dir_bool:

		for pdb_file in pdb_files:

			## Get the protein information dataframe 

			protein_df = get_protein_information(pdb_file, input_dir)

			## Get the SASA for the protein

			calculate_sasa_protein(protein_df, new_params)

			res_sasa = calculate_residue_sasa_protein(protein_df)

			## Get the SASA for the solvated dipeptides

			sol_sasa = calculate_residue_sasa_solvent(protein_df, new_params)

			## Get the rSASA

			rsasa_vals = res_sasa/sol_sasa

			## Create dataframe and organize easily

			rsasa_df = pd.DataFrame()

			rsasa_df["residue_name"] = get_aa_list(protein_df)

			rsasa_df["SASA_protein"] = res_sasa 

			rsasa_df["SASA_dipep"] = sol_sasa

			rsasa_df["rSASA"] = rsasa_vals

			## Save the data to a file

			os.chdir(output_dir)

			if output_csv_bool:

				save_filename = pdb_file.split(".pd")[0] + "_sasa_data.csv"

				rsasa_df.to_csv(save_filename)

			elif not output_csv_bool:

				save_filename = pdb_file.split(".pd")[0] + "_sasa_data.xlsx"

				rsasa_df.to_excel(save_filename)



if __name__ == '__main__':
	main()







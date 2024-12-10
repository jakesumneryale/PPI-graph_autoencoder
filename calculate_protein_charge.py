import numpy as np 
import pandas as pd
import os

## Local imports

from create_protein_graph_structure import *
from standalone_freesasa_rsasa_code import *

def calculate_interface_amino_acids_rsasa(protein_df, cutoff = 1e-6):
	'''
	Calculates the interface amino acids for a given 
	protein heterodimer by calculating the rSASA
	of the dimer and both of the monomers. Then it will
	return the residues based on whether the delta rSASA
	is greater than the cutoff
	'''

	## Get both of the chains

	chain1_df = protein_df[protein_df["chain_id"] == 1]
	chain2_df = protein_df[protein_df["chain_id"] == 2]


	## Calculate the rSASA for the entire heterodimer and the monomer chains

	complex_rsasa = calculate_rsasa_for_protein(protein_df)
	chain1_rsasa = calculate_rsasa_for_protein(chain1_df)
	chain2_rsasa = calculate_rsasa_for_protein(chain2_df)

	monomer_rsasa = pd.concat([chain1_rsasa, chain2_rsasa]) ## Same order as the original protein_df

	## Return the IDs of the interface amino acids (the list of the amino_acid_id)

	all_inds = np.where(np.abs(monomer_rsasa["rSASA"] - complex_rsasa["rSASA"]) > cutoff)



##Create Graph Format
##Naomi Brandt
## 2/10/25

from pathlib import Path
import numpy as np
import h5py
import argparse
import pandas as pd
import re
from scipy.stats import spearmanr, pearsonr
from os import listdir
from os.path import isfile, join
import create_protein_graph_structure_jake as jk
import Bounded_Voronoi_Contacts_noargs_radical as voro

def initialize_graphs(pdb_id,pdb_dir,save_dir = "./", file_indicator = "_H_0001.pdb"):
    
    graph_fh=Path(save_dir) / Path(f'{pdb_id}.hdf5')
    fh=h5py.File(str(graph_fh),'w')
    sdt = h5py.string_dtype(encoding='utf-8')

    all_decoys = sorted([f for f in listdir(pdb_dir) if isfile(join(pdb_dir, f)) and file_indicator in f])
    
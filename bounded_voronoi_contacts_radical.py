##bounded_voronoi_contacts_radical.py
## Naomi Brandt
## Created 1/3/25
## Edited 1/21/25 - Added radical tessellation

import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, isdir
from numpy.linalg import norm
from pathlib import Path
import argparse
from glob import glob 
import h5py
import pyvista as pv
import pyvoro
import create_protein_graph_structure as jk
import trimesh as tm


parser = argparse.ArgumentParser()
parser.add_argument('-d','--dir',help='Directory to file')
parser.add_argument('-f','--fh',help='File name of pdb')
parser.add_argument('-p','--probe',default=1.4,help='Probe size to define boundary')
parser.add_argument('-o','--out',required=False,help='Output file name indicator, must be included if save=True')
parser.add_argument('-od','--out_dir',required=False,help='Output file directory, must be included if save=True')
parser.add_argument('-s','--save', default=False,help='Save outputs to .csv files')
args=parser.parse_args()

aa_one_to_three={v: k for k, v in jk.aa_three_to_one.items()}


def create_surface(coords,atomic_radii,probe_size=1.4):

    '''
    Creates a point cloud about the surface of protein by merging icospheres
    Point cloud must be generated at atomic radius+2*probe radius+point radius (currently set to .001)
    to generate Voronoi boundaries at desired location

    Input
    coords: Center coordinates of each atom in protein in [x,y,z] format
    atomic_radii: Associated radii per atom
    probe_size: Desired probe size, should be half of maximum envagination

    Output
    combined: original set of vertices + set of vertices making up the surface point cloud
    '''
    multi_spheres=[]
    for i,pt in enumerate(coords):
        ##Uses pyvista to make set of 162 pt icosphere centered on atoms
        sphere=pv.Icosphere(
        radius=atomic_radii[i]+2*probe_size+.001,
        center=pt,
        nsub=2)

        ##Turn spheres to trimesh format
        sphere_as_array = sphere.faces.reshape((sphere.n_faces, 4))[:, 1:] 
        sphere_tmesh=tm.Trimesh(sphere.points, sphere_as_array) 
        multi_spheres.append(sphere_tmesh)

    ## Take intersection and union of set of spheres, returns difference of union & intersection
    inter=tm.boolean.intersection(multi_spheres,engine='manifold')
    union=tm.boolean.union(multi_spheres,engine='manifold')

    combined=np.empty((0,3))
    for vert in union.vertices:
        if (vert == inter.vertices).all(axis=1).any(axis=0):
            continue
        else:
            combined=np.vstack((combined,vert))
            
    return combined


def get_bounded_voro(protein_df, box_margin = 1, dispersion = 4.5,probe_size = 1.4):
    '''
    Edited from create_protein_graph_structure_jake
    Gets the voronoi tesselation of the coordinates
    that are specified in the protein_df. 
    
    Input
    protein_df: protein dataframe in format from jk.get_protein_information()
    box_margin: padding added to box volume
    dispersion: grid size for voronoi calculation
    probe_size: see create_surface()

    Output
    clean_bounded_voro: voronoi tessellation which only includes adjacencies/cells
    that are not part of the surface point cloud
    '''
    
    ## Get the protein coordinates
    
    protein_coords = np.zeros((len(protein_df), 3))
    
    protein_coords[:, 0] = protein_df["x_coord"]
    protein_coords[:, 1] = protein_df["y_coord"]
    protein_coords[:, 2] = protein_df["z_coord"]
    
    ## Shift the coordinates to be centered at the origin
    
    protein_coords -= np.mean(protein_coords, axis = 0)
    
    ## Get atomic radii list
    
    atomic_radii = np.array(protein_df["atom_radius"])
    
    ## Calculate the surface with given probe
    
    surface=create_surface(protein_coords,atomic_radii,probe_size)
    
    all_coords=np.vstack((protein_coords,surface))
    
    surface_rad=np.full((len(surface),),.001)
    all_rad=np.concatenate((atomic_radii,surface_rad))
    
    ## Minimum box size that works
    
    lim_x = [np.min(all_coords[:, 0]) - box_margin, np.max(all_coords[:, 0]) + box_margin]
    lim_y = [np.min(all_coords[:, 1]) - box_margin, np.max(all_coords[:, 1]) + box_margin]
    lim_z = [np.min(all_coords[:, 2]) - box_margin, np.max(all_coords[:, 2]) + box_margin]
    
    box_lims = [lim_x, lim_y, lim_z]

    ## Compute the tesselation

    voronoi_tessellation = pyvoro.compute_voronoi(all_coords, box_lims, dispersion,all_rad)


    ## Remove cells and faces from voronoi_tessellation that are part of the surface boundary
    clean_bounded_voro=[]
    real_cells=[]
    for ind,cell in enumerate(voronoi_tessellation):
        if (cell['original'] == protein_coords).all(axis=1).any(axis=0):
            cell['cell_id']=ind
            clean_bounded_voro.append(cell.copy())
            real_cells.append(ind)
  
    for cell in clean_bounded_voro:
        cell['faces']=[face for face in cell['faces'] if face['adjacent_cell'] in real_cells]
    
    return clean_bounded_voro

def get_all_contacts_aa(protein_df,voronoi_tessellation):
    '''
    Calculates the voronoi neighbors at the amino acid level using jk.get_voronoi_neighbors_aa
    Generates dataframe of all unique unique interactions (inter and intrachain)
    
    Input
    protein_df: protein dataframe in format from jk.get_protein_information()
    voronoi_tessellation: voronoi tessellation (from either get_bounded_voro or pyvoro.compute_voronoi())

    Output
    all_contacts: dataframe in which each row represents a contact, with information including
    index of each participating amino acid, residue type in 3-letter code, and chain id
    '''
    
    protein_neighbor_mat=jk.get_voronoi_neighbors_aa(protein_df, voronoi_tessellation)

    all_contacts=np.argwhere(protein_neighbor_mat==1)
    all_contacts_df=pd.DataFrame(columns=['aa_id1','index1','restype1','chain1','aa_id2','index2','restype2','chain2'])

    
    contact_set_list=[]

    for ind,contact in enumerate(all_contacts):
            
        aai=contact[0]
        aaj=contact[1]

        contact_set={aai,aaj}

        if contact_set not in contact_set_list:

            contact_set_list.append(contact_set)

            chaini=protein_df[protein_df['aa_id']==aai]['chain_id'].iloc[0]
            chainj=protein_df[protein_df['aa_id']==aaj]['chain_id'].iloc[0]


            rec_ind=protein_df[protein_df['aa_id']==aai]['aa_ind'].iloc[0]
            rec_type=aa_one_to_three[protein_df[protein_df['aa_id']==aai]['aa_name'].iloc[0]]
            lig_ind=protein_df[protein_df['aa_id']==aaj]['aa_ind'].iloc[0]
            lig_type=aa_one_to_three[protein_df[protein_df['aa_id']==aaj]['aa_name'].iloc[0]]

            contact_data=[aai,rec_ind,rec_type,chaini,aaj,lig_ind,lig_type,chainj]
            contact_df=pd.DataFrame(contact_data).T
            contact_df.columns=all_contacts_df.columns
            all_contacts_df=pd.concat([all_contacts_df,contact_df])

    all_contacts_df.reset_index(inplace=True,drop=True)
    return all_contacts_df


def get_interface_contacts_aa(protein_df,voronoi_tessellation):
    '''
    Calculates the voronoi neighbors at the amino acid level using jk.get_voronoi_neighbors_aa
    Generates dataframe of all unique interchain interactions at the amino acid level
    (Only works for 2 chain interfaces)
    
    Input
    protein_df: protein dataframe in format from jk.get_protein_information()
    voronoi_tessellation: voronoi tessellation (from either get_bounded_voro or pyvoro.compute_voronoi())

    Output
    interface_contacts: dataframe in which each row represents a contact, with information including
    index of each participating amino acid, residue type in 3-letter code, and chain id
    '''
    protein_neighbor_mat=jk.get_voronoi_neighbors_aa(protein_df, voronoi_tessellation)

    all_contacts=np.argwhere(protein_neighbor_mat==1)

    interface_contacts=pd.DataFrame(columns=['receptor_index','receptor_restype','receptor_chain','ligand_index','ligand_restype','ligand_chain',])


    for ind,contact in enumerate(all_contacts):
            
        ## Looks where neighbor matrix == 1, checks if contacts are in two different chains, adds to df if so
        aai=contact[0]
        aaj=contact[1]

        chaini=protein_df[protein_df['aa_id']==aai]['chain_id'].iloc[0]
        chainj=protein_df[protein_df['aa_id']==aaj]['chain_id'].iloc[0]

        if ind==0:
            chain1=protein_df[protein_df['aa_id']==aai]['chain_id'].iloc[0]

        ## Stops once it gets to the second chain (note that this only works for 2 chain interfaces)
        if chaini != chain1:
            break

        if chaini != chainj:

            rec_ind=protein_df[protein_df['aa_id']==aai]['aa_ind'].iloc[0]
            rec_type=aa_one_to_three[protein_df[protein_df['aa_id']==aai]['aa_name'].iloc[0]]
            lig_ind=protein_df[protein_df['aa_id']==aaj]['aa_ind'].iloc[0]
            lig_type=aa_one_to_three[protein_df[protein_df['aa_id']==aaj]['aa_name'].iloc[0]]

            contact_data=[rec_ind,rec_type,chaini,lig_ind,lig_type,chainj]
            contact_df=pd.DataFrame(contact_data).T
            contact_df.columns=interface_contacts.columns
            interface_contacts=pd.concat([interface_contacts,contact_df])

    interface_contacts.reset_index(inplace=True,drop=True)

    return interface_contacts

def get_voronoi_neighbors_atom(protein_df, voronoi_tessellation):
    '''
    Gets the voronoi neighbors of all atoms in the protein
    using the voronoi neighbors as the metric by which neighbors are
    determined. Output is an adjacency matrix of the voronoi
    neighbors
    '''
    
    ## Get total protein aa count
    
    total_prot_len = np.max(protein_df.index)+1
    
    ## Init neighbor adjacency matrix
    
    neighbor_adj_mat_atom= np.zeros((total_prot_len, total_prot_len))

    ## Loop through the atoms and add to the adjacency matrix
    
    for i in range(total_prot_len):
               
        ## Loop through all the atoms in the protein
        
        atom_neighbor_set = set([])
        voro_cell = voronoi_tessellation[i]
        neighs = set([f["adjacent_cell"] for f in voro_cell["faces"]])
        atom_neighbor_set |= neighs
            
        ## Link the atom neighbors to specific amino acids and add to the adj mat
        
        for j in atom_neighbor_set:   
            if j < 0:
                continue
                ## odd thing with getting negative values cell IDs? 
            
            ## Update adj mat
            neighbor_adj_mat_atom[i, j] = 1
            
    return neighbor_adj_mat_atom

def add_packing_info(protein_df,neighbor_adj_mat_atom,neighbor_adj_mat_aa,voronoi_tessellation,rSASA_path):
    protein_df['cell_volume']=np.zeros(len(protein_df))
    protein_df['atom_neighbors']=sum(neighbor_adj_mat_atom)
    protein_df['aa_neighbors']=np.zeros(len(protein_df))

    for ind,cell in enumerate(voronoi_tessellation):
        protein_df.at[ind,'cell_volume']=cell['volume']

    for ind,val in enumerate(sum(neighbor_adj_mat_aa)):
        aa_inds=protein_df[protein_df['aa_id']==ind].index
        for i in aa_inds:
            protein_df.at[i,'aa_neighbors']=val
    
    dimer_rsasa_df=pd.read_csv(rSASA_path)
    protein_df['rSASA']=np.zeros(len(protein_df))

    for i in dimer_rsasa_df['residue_ind']:
        aa_inds=protein_df[protein_df['aa_ind']==i].index
        for j in aa_inds:
            protein_df.at[j,'rSASA']=dimer_rsasa_df[dimer_rsasa_df['residue_ind']==i].iloc[0]['rSASA']

    return protein_df



def main():

    pdb_name=args.fh
    pdb_dir=Path(args.dir)
    probe_size=args.probe
    out_indicator=args.out
    out_dir=Path(args.out_dir)
    save=args.save
        
    protein_df=jk.get_protein_information(pdb_name,pdb_dir)
    
    bounded_voro_tessellation=get_bounded_voro(protein_df,box_margin=1,dispersion=4.5,probe_size=probe_size)
    all_contacts=get_all_contacts_aa(protein_df=protein_df,voronoi_tessellation=bounded_voro_tessellation)
    interface_contacts=get_interface_contacts_aa(protein_df=protein_df,voronoi_tessellation=bounded_voro_tessellation)

    neighbor_adj_mat_aa=jk.get_voronoi_neighbors_aa(protein_df, bounded_voro_tessellation)
    neighbor_adj_mat_atom=get_voronoi_neighbors_atom(protein_df, bounded_voro_tessellation)

    protein_df=add_packing_info(protein_df,neighbor_adj_mat_atom,neighbor_adj_mat_aa,bounded_voro_tessellation,rSASA_path)

    if save:

        all_contacts_out_fh=f'{pdb_name}_all_contacts.csv'
        interface_contacts_out_fh=f'{pdb_name}_interface_contacts.csv'
        protein_out_fh=f'{pdb_name}_protein_data.csv'

        save_dir=out_dir
        out_dir.mkdir(exist_ok=True)

        all_contacts.to_csv(save_dir / Path(all_contacts_out_fh))
        interface_contacts.to_csv(save_dir / Path(interface_contacts_out_fh))
        protein_df.to_csv(save_dir / Path(protein_out_fh))







if __name__ == '__main__':
	main()

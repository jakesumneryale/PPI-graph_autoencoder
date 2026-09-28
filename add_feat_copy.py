from pathlib import Path
import numpy as np
import h5py
import argparse
import pandas as pd
import re
from scipy.stats import spearmanr, pearsonr
from os import listdir
from os.path import isfile, join
import create_protein_graph_structure as jk
import bounded_voronoi_contacts_radical as voro
from numpy.linalg import norm
import pickle
import torch
from torch_geometric import datasets
from Dataset import graph_dataset
from numpy.linalg import norm
import os

aa_one_to_three={v: k for k, v in jk.aa_three_to_one.items()}
sdt = h5py.string_dtype(encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('-p','--pdb',help='pdbid')
args=parser.parse_args()

def calc_rsasa(pdbid):
    decoy_dir=Path(f'/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/balanced_decoys/{pdbid}_balanced')
    #decoy_dir=Path('/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/targets')
    os.chdir(decoy_dir)
    rSASA_dir=decoy_dir / Path(f'{pdbid}_rSASA_temp')
    save_dir= Path(f'/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/rSASA/{pdbid}_rSASA')
    rSASA_dir.mkdir(exist_ok=True)
    save_dir.mkdir(exist_ok=True)

    avg_rSASA=pd.read_csv('/gpfs/gibbs/pi/ohern/nb685/New_GNN/norm_rSASA.csv')

    oneletter=[jk.aa_three_to_one[i] for i in avg_rSASA['Residue'].values]
    avg_rSASA_dict={key:value for key,value in zip(oneletter,avg_rSASA['avg_rSASA'].values)}

    file_indicator=".pdb"

    all_decoys = sorted([f for f in listdir(decoy_dir) if isfile(join(decoy_dir, f)) and file_indicator in f])
    run_decoys_csvs=sorted([f for f in listdir(save_dir) if isfile(join(save_dir, f))])
    run_decoys=[i.split('.csv')[0] for i in run_decoys_csvs]
    for decoy in all_decoys:
        os.chdir(decoy_dir)

        decoy_name = decoy.split(file_indicator)[0]
        if decoy_name in run_decoys:
            continue
        else:
            os.system(f'cp {decoy} {rSASA_dir}')
            os.chdir(rSASA_dir)
            os.system(f'pdb_splitchain.py {decoy}')
            os.system('rm -- *\ *')

            decoy_files=sorted([f for f in listdir(rSASA_dir) if isfile(join(rSASA_dir, f)) and decoy_name in f])

            chain=0
            for file in decoy_files:
                protein_df=jk.get_protein_information(file,rSASA_dir)

                if file==decoy:
                    complex_df=jk.calculate_rsasa_for_protein(protein_df)
                    
                elif chain==0:
                    rec_df=jk.calculate_rsasa_for_protein(protein_df)
                    chain+=1

                elif chain==1:
                    lig_df=jk.calculate_rsasa_for_protein(protein_df)

                else:
                    print(f'Too many files/chains for complex {decoy_name}!')
                        
            monomer_df=pd.concat([rec_df,lig_df])
            final_df=complex_df.merge(monomer_df, on=['residue_ind','residue_name','chain_id'],suffixes=['_complex','_monomer'])
            final_df['delta_rSASA']=final_df['rSASA_complex']-final_df['rSASA_monomer']
            
            delta_R_complex=[]
            delta_R_monomer=[]
            for i in final_df.index:
                res=final_df.loc[i]['residue_name']
                delta_R_complex.append((final_df.loc[i]['rSASA_complex']-avg_rSASA_dict[res])/avg_rSASA_dict[res])
                delta_R_monomer.append((final_df.loc[i]['rSASA_monomer']-avg_rSASA_dict[res])/avg_rSASA_dict[res])

            final_df['drSASA_complex']=delta_R_complex
            final_df['drSASA_monomer']=delta_R_monomer
            
            final_df.to_csv(save_dir / Path(f'{decoy_name}.csv'))
            
            os.system(f'rm {decoy_name}*')

    os.chdir(decoy_dir)
    os.system(f'rm {rSASA_dir} -r')

def compile_rsasa(pdb):
    rsasa_dir=Path(f'/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/rSASA')

    target=pdb
    target_rsasa=rsasa_dir / Path(f'{pdb}_rSASA')
    rsasa_csvs=sorted([f for f in listdir(target_rsasa) if isfile(join(target_rsasa,f))])
    all_rand_d_rsasa=pd.DataFrame(columns=['residue_name','drSASA_complex','drSASA_monomer','delta_rSASA'])
    all_samp_d_rsasa=pd.DataFrame(columns=['residue_name','drSASA_complex','drSASA_monomer','delta_rSASA'])


    for csv in rsasa_csvs:
        csv_fh=target_rsasa / Path(csv)
        rsasa_df=pd.read_csv(csv_fh)
        decoy_df=rsasa_df[['residue_name','drSASA_complex','drSASA_monomer','delta_rSASA']]
        if 'random' in csv:
            all_rand_d_rsasa=pd.concat([all_rand_d_rsasa,decoy_df],ignore_index=True)
        elif 'sampled' in csv:
            all_samp_d_rsasa=pd.concat([all_samp_d_rsasa,decoy_df],ignore_index=True)
        else:
            print(csv)
    

    all_rand_d_rsasa.to_csv(rsasa_dir / Path('drSASA',f'{target}_random_all_drSASA.csv'))
    all_samp_d_rsasa.to_csv(rsasa_dir / Path('drSASA',f'{target}_sampled_all_drSASA.csv'))


def add_contacts(pdb):
    pdb_dir=Path(f'/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/balanced_decoys/{pdb}_balanced')
    graph_dir=Path(f'/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/balanced_graphs/{pdb}.hdf5')
    with h5py.File(graph_dir,'r+') as fh:
        for ind,decoy in enumerate(fh.keys()):

            g=fh[decoy]

            try:
                contacts=g['edge_features']['contacts'][()]
                ref=g['edge_reference'][()]
                protein_df=jk.get_protein_information(f'{decoy}.pdb',pdb_dir)
                ca_dist=np.empty((len(contacts)))

                for ind,contact in enumerate(contacts):

                    aai=contact[0]
                    aaj=contact[1]
                    rec_ca_coords=protein_df[(protein_df['aa_id']==aai) & (protein_df['atom_name']=='CA')][['x_coord','y_coord','z_coord']].values
                    lig_ca_coords=protein_df[(protein_df['aa_id']==aaj) & (protein_df['atom_name']=='CA')][['x_coord','y_coord','z_coord']].values
                    ca_dist[ind]=np.sqrt((rec_ca_coords[0][0]-lig_ca_coords[0][0])**2+(rec_ca_coords[0][1]-lig_ca_coords[0][1])**2+(rec_ca_coords[0][2]-lig_ca_coords[0][2])**2)
                g['edge_features'].create_dataset('ca_dist',data=ca_dist)

            except Exception as error:
                print(decoy,error)
                   


def calc_avg_drsasa(pdb):
    rsasa_dir=Path(f'/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/rSASA')
    ss_data_dir=Path('/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/all_supersampled_balanced_csv')

    ss_data_csvs=sorted([f for f in listdir(ss_data_dir) if isfile(join(ss_data_dir,f)) and '.csv' in f])





    csv_fh=ss_data_dir / Path(f'{pdb}_supersampled_balanced_scores.csv')
    ss_data_df=pd.read_csv(csv_fh)
        
    targ=pdb

    rsasa_targ_dir=rsasa_dir/Path(f'{targ}_rSASA')

    
    avg_d_rsasa=[]
    
    for decoy in ss_data_df['Decoy'].values:
        try:
            rsasa_csv=rsasa_targ_dir / Path(f'{decoy}.csv')
            rsasa_df=pd.read_csv(rsasa_csv)
            d_rsasa=rsasa_df['drSASA_complex'].values
            avg_d_rsasa.append(np.mean(d_rsasa))
        except:
            print(f'File not found: {decoy}')
    
    d_rsasa_df=pd.DataFrame(zip(ss_data_df['Decoy'].values,avg_d_rsasa),columns=['Decoy','avg_d_rsasa'])
    d_rsasa_df=pd.merge_ordered(d_rsasa_df,ss_data_df[['Decoy','DockQ']], on='Decoy')
    d_rsasa_df.to_csv(rsasa_targ_dir / Path(f'{targ}_avg_dRSASA.csv'),index=False)

def main():
    # add_contacts(args.pdb)
    calc_rsasa(args.pdb)
    compile_rsasa(args.pdb)
    calc_avg_drsasa(args.pdb)

if __name__ == '__main__':
	main()
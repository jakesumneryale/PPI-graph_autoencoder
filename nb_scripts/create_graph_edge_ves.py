from pathlib import Path
import numpy as np
import h5py
import argparse
import pandas as pd
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings("ignore")

argParser=argparse.ArgumentParser()

q_e=1.6*10**-19
e_0=8.85*10**-12
D=80
sig=4.8*10**-10
ep_es=q_e**2/(4*np.pi*e_0*D*sig)
lam=9*10**-10
k_b=1.38*10**-23
T=293

def main():

    
    argParser.add_argument('-p','--pdbid')
    args=argParser.parse_args()

    pdbid=args.pdbid

    save_dir=Path('/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/decoy_interface_edge_charge')
    complex_dir=Path(f'/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/balanced_decoys/{pdbid}_balanced/{pdbid}_partial_charges')
    graph_fh=Path(f'/gpfs/gibbs/pi/ohern/nb685/Decoys/Balanced_Dataset/balanced_graphs/{pdbid}.hdf5')

    interface_edges={}
    edge_reference={}
    interface_nodes={}
    with h5py.File(str(graph_fh),'r') as fh:
        for key in fh.keys():
            g=fh[key]
            try:
                interface_edges[key]=g['edge_features']['interface_edges'][()]
                edge_reference[key]=g['edge_features']['contacts'][()]
                interface_nodes[key]=g['node_features']['interface_nodes'][()]
            except Exception as error:
                print(error,flush=True)
                continue

    all_ves_df=pd.DataFrame(columns=['Edge_Ves'],index=list(interface_edges.keys()))
    for i,key in enumerate(interface_edges.keys()):
        interface_edge_ref=edge_reference[key][np.where(interface_edges[key] !=0)[0]]

        if key.startswith('complex'):
            file_indicator='_corrected_H_0001_Q.pqr'
        else:
            file_indicator='_Q.pqr'

        fh=key+file_indicator

        try:


            pqr_df=pd.read_csv(complex_dir/ Path(fh),header=None,names=['Field_name','Atom_number','Atom_name','Residue_name','Residue_number','X','Y','Z','Charge','Radius'],delimiter=r"\s+",encoding_errors='replace')
            pqr_df=pqr_df[(pqr_df['Field_name']!='TER') & (pqr_df['Field_name']!='END')]


            res_num=pqr_df['Residue_number'].values

            unique, counts = np.unique(res_num, return_counts=True)
            Na=dict(zip(unique, counts))

            atom_interface_bool=[]
            res_indexing=[]
            for i,key1 in enumerate(Na.keys()):
                atom_interface_bool+=[interface_nodes[key][i]]*Na[key1]
                res_indexing+=[i]*Na[key1]

            pqr_df['interface_bool']=atom_interface_bool
            pqr_df['res_index']=res_indexing

            pqr_inter_df=pqr_df[pqr_df['interface_bool']!=0].copy()

            coords=pqr_inter_df[['X','Y','Z']].values

            dist_mat=distance_matrix(coords,coords)*10**-10
            dist_mat[dist_mat == 0] = np.nan


            q1=np.matrix(pqr_inter_df['Charge'].values).T
            q2=np.matrix(pqr_inter_df['Charge'].values)

            charge_mat=np.array(q1*q2)
            pot_mat=charge_mat*sig/dist_mat*np.exp(-dist_mat/lam)

            inter_res_num=pqr_inter_df['Residue_number'].values
            unique, counts = np.unique(inter_res_num, return_counts=True)
            Na_inter=dict(zip(unique, counts))

            aa_interact=np.zeros((len(Na_inter),len(Na_inter)))
            count1=0

            for i,key1 in enumerate(Na_inter.keys()):
                row_start=count1
                row_end=count1+Na_inter[key1]
                count1=count1+Na_inter[key1]
                count2=count1
                for j,key2 in enumerate(Na_inter.keys()):
                    if int(key1)<int(key2):
                        count2=count2+Na_inter[key2]
                        col_start=count2-Na_inter[key2]
                        col_end=count2
                        aa_interact[i,j]=np.sum(pot_mat[row_start:row_end,col_start:col_end])
                        aa_interact[j,i]=np.sum(pot_mat[row_start:row_end,col_start:col_end])


            inter_res_indexing=pqr_inter_df[pqr_inter_df['Atom_name']=='CA']['res_index'].values
            reindex_dict={key:value for key,value in zip(inter_res_indexing,range(len(inter_res_indexing)))}
            reindex_interface_edge_ref=[[reindex_dict[i],reindex_dict[j]] for i,j in interface_edge_ref]
            interface_interact=[]
            for i in reindex_interface_edge_ref:
                interface_interact.append(aa_interact[i[0],i[1]])

            interface_edge_dict={tuple(key):value for key, value in zip(interface_edge_ref,interface_interact)}

            all_edge_ves=[]

            for i in edge_reference[key]:
                if (i[0],i[1]) in interface_edge_dict.keys():
                    all_edge_ves.append((interface_edge_dict[(i[0],i[1])]))
                else:
                    all_edge_ves.append(0)

            all_ves_df.at[key,'Edge_Ves']=all_edge_ves
        except Exception as error:
            print(error)
            continue

    all_ves_df.to_csv(save_dir / Path(f'{pdbid}_edge_ves.csv'))
    all_ves_df.to_pickle(save_dir/ Path(f'{pdbid}_edge_ves.pkl'))

if __name__ == '__main__':
	main()
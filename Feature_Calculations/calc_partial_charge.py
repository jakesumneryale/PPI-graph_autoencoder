from pathlib import Path
import argparse
from os import listdir
from os.path import isfile, join
import os


argParser=argparse.ArgumentParser()

def main():
    argParser.add_argument('-p','--pdbid')
    argParser.add_argument('-d','--dir',help='Complex Directory')
    args=argParser.parse_args()

    complex_dir=Path(args.dir)
    savedir=complex_dir / Path(f'{args.pdbid}_partial_charges')
    savedir.mkdir(exist_ok=True)
    all_targs=sorted([f for f in listdir(complex_dir) if isfile(join(complex_dir,f)) and '.pdb' in f])

    for target_fh in all_targs:
        fh=complex_dir / Path(target_fh)
        targ=target_fh[:-4]
        pqr_fh=savedir / Path(f'{targ}_Q.pqr')
        pdb_fh=savedir / Path(f'{targ}_Q.pdb')
        os.system(f'pdb2pqr {fh} {pqr_fh} --ff PARSE --pdb-output {pdb_fh} --whitespace')


if __name__ == '__main__':
	main()

"""Generate APBS/pdb2pqr surface electrostatic potential maps from PPI PDB models.

Deliberately standalone from the graph HDF5 files: this package only *reads*
PDB structures and writes its own per-target HDF5 stores, so electrostatics can
be regenerated or reanalysed without touching the training graphs.
"""

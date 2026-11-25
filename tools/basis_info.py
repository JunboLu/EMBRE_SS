#! /usr/env/bin python

import numpy as np

def get_atom_ao_indices(atom_list, sup_slices):

  idx = []
  for ia in atom_list:
    p0 = sup_slices[ia, 2]
    p1 = sup_slices[ia, 3]
    idx.extend(range(p0, p1))

  return np.asarray(idx, dtype=int)

def get_sup_ao_indices(mol_sup):

  return mol_sup.aoslice_by_atom()



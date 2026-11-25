#! /usr/env/bin/python

import copy
import numpy as np
from pyscf import lib
from pyscf import dft
from scipy import special
from scipy.optimize import linear_sum_assignment
from EMBRE_SS.tools import gen_mol

einsum = np.einsum
cos = np.cos
sin = np.sin
norm = np.linalg.norm
erf = special.erf
erfc = special.erfc

def get_eta(atoms, qm_indices, mm_indices):

  ele2radius = {
    'La': 1.356, 'Ce': 1.336, 'Pr': 1.319, 'Nd': 1.303, 'Pm': 1.284,
    'Sm': 1.272, 'Eu': 1.260, 'Gd': 1.247, 'Tb': 1.235, 'Dy': 1.223,
    'Ho': 1.212, 'Er': 1.202, 'Tm': 1.192, 'Yb': 1.182, 'Lu': 1.172,
    'Y': 1.215,
    'F': 1.16, 'Cl': 1.67, 'Br': 1.82,
    'O': 1.22,
    }

  eta = lib.param.BOHR / np.asarray([ele2radius[a[0]] for a in atoms])
  qm_eta = eta[qm_indices]
  mm_eta = eta[mm_indices]

  return eta, qm_eta, mm_eta  

def get_iGvR(Gv, coords, charges=None):

  GvR = einsum('gx,ix->ig', Gv, coords)
  cosGvR = cos(GvR)
  sinGvR = sin(GvR)

  if charges is not None:
    zcosGvR = einsum('i,ig->g', charges, cosGvR)
    zsinGvR = einsum('i,ig->g', charges, sinGvR)
  else:
    zcosGvR = zsinGvR = None
  return cosGvR, sinGvR, zcosGvR, zsinGvR

def rs_sum(coords1, coords2, charges2, f):
  rij = coords1[:,None,:] - coords2[None]
  rij = norm(rij, axis=-1)
  return einsum('ij,j->i', f(rij), charges2)

def mm_hcore(mol_sup, atoms, charges, cell_vector, qm_indices, mm_indices, work_dir):

  eta, qm_eta, mm_eta = get_eta(atoms, qm_indices, mm_indices)

  coords = np.asarray([a[1] for a in atoms]) / lib.param.BOHR
  qm_coords = coords[qm_indices]
  mm_coords = coords[mm_indices]

  qm_charges = charges[qm_indices]
  mm_charges = charges[mm_indices]

  mol = copy.deepcopy(mol_sup)
  #mol.charge = 1
  #mol.build()

  cell = gen_mol.get_cell(mol, cell_vector)

  ew_eta, ew_cut = cell.get_ewald_params()
  log_precision = np.log(cell.precision / (np.abs(mm_charges).sum()*16*np.pi**2))
  ke_cutoff = -2*ew_eta**2*log_precision
  mesh = cell.cutoff_to_mesh(ke_cutoff)
  Lall = np.asarray(cell.get_lattice_Ls(rcut=ew_cut))
  Gv, Gvbase, Gweights = cell.get_Gv_weights(mesh)
  Gv = np.asarray(Gv)
  absG2 = einsum('gx,gx->g', Gv, Gv)
  absG2[absG2==0] = 1e200
  coulG = 4 * np.pi / absG2
  coulG *= Gweights
  Gpref = np.exp(-absG2/(4*ew_eta**2)) * coulG

  mf = dft.RKS(mol)
  mf.grids.build()

  cosGvR, sinGvR, zcosGvR, zsinGvR = \
    get_iGvR(Gv, coords, charges)
  v_ao = 0

  prog = 0
  for ao, mask, weight, grid_coords in mf._numint.block_loop(mol, mf.grids, blksize=1680):
    prog += grid_coords.shape[0]
    v = np.zeros(len(grid_coords))
    grid_coords = np.asarray(grid_coords)
    weight = np.asarray(weight)

    # ewald real-space
    for T in Lall:
        # pot on grids from MM charges
        mm_coords_ = mm_coords + T
        v += rs_sum(grid_coords, mm_coords_, mm_charges,
                lambda r: (erf(mm_eta * r) - erf(ew_eta * r)) / r)
        # pot on grids from QM images
        if norm(T) < 1e-10:
            v -= rs_sum(grid_coords, qm_coords, qm_charges,
                    lambda r: erf(ew_eta * r) / r)
        else:
            qm_coords_ = qm_coords + T
            v += rs_sum(grid_coords, qm_coords_, qm_charges,
                    lambda r: (erf(qm_eta * r) - erf(ew_eta * r)) / r)

    # ewald g-space
    cosGvRg, sinGvRg, _, _ = get_iGvR(Gv, grid_coords)
    v += einsum('ig,g,g->i', cosGvRg, zcosGvR, Gpref)
    v += einsum('ig,g,g->i', sinGvRg, zsinGvR, Gpref)

    # represent v on AO basis
    v *= weight
    ao = np.asarray(ao)
    aowv = einsum('g,gi->gi', v, ao)
    v_ao += einsum('gi,gj->ij', aowv, ao)

  mm_hcore_file = ''.join((work_dir, '/mm_hcore.npy'))
  np.save(mm_hcore_file, -v_ao)

  return mm_hcore_file

#! /usr/env/bin python

import uuid
import subprocess
import numpy as np
from scipy.linalg import eigh
from pyscf import df
from pyscf import scf
from EMBRE_SS.main import myavas
from EMBRE_SS.tools import re_info
from EMBRE_SS.tools import basis_info
from EMBRE_SS.tools import read_coord
from EMBRE_SS.main import cahf
from EMBRE_SS.main import subsystem
from EMBRE_SS.main import cahf_init_guess

def make_sup_density(nao_sup, a_ao_index, b_ao_index, D_a, D_b):

  D_sup_alpha = np.zeros((nao_sup, nao_sup))
  D_sup_beta = np.zeros((nao_sup, nao_sup))

  D_sup_alpha[np.ix_(a_ao_index, a_ao_index)] += D_a[0]
  D_sup_alpha[np.ix_(b_ao_index, b_ao_index)] += D_b[0]

  D_sup_beta[np.ix_(a_ao_index, a_ao_index)] += D_a[1]
  D_sup_beta[np.ix_(b_ao_index, b_ao_index)] += D_b[1]

  return D_sup_alpha, D_sup_beta

def embed_fragment_mos(C_frag, ao_indices, nao_sup):

  C_sup = np.zeros((nao_sup, C_frag.shape[1]), dtype=C_frag.dtype)
  C_sup[ao_indices, :] = C_frag
  return C_sup


def metric_orthogonalize(C, S, against=None, lindep=1e-9):

  X = C.copy()
  if against is not None and against.shape[1] > 0:
    X -= against @ (against.conj().T @ S @ X)

  M = X.conj().T @ S @ X
  M = (M + M.conj().T) * 0.5
  w, U = eigh(M)
  keep = w > lindep

  if not np.all(keep):
    raise RuntimeError(
      'Fragment orbitals become linearly dependent during S-orthogonalization: '
      f'{np.count_nonzero(keep)}/{len(w)} independent orbitals. '
      'Check fragment AO mapping and inter-fragment overlap.'
    )

  X = X @ U @ np.diag(w**-0.5) @ U.conj().T
  return X

def build_fragment_mo_set(mol_sup, mf_sup, C_a, occ_a, C_b, occ_b,
                          a_ao_indices, b_ao_indices, dm_sup):

  nao_sup = mol_sup.nao_nr()
  Ca = embed_fragment_mos(C_a, a_ao_indices, nao_sup)
  Cb = embed_fragment_mos(C_b, b_ao_indices, nao_sup)

  C_all = np.hstack((Ca, Cb))
  occ_all = np.concatenate((np.asarray(occ_a), np.asarray(occ_b)))

  if C_all.shape[1] != nao_sup:
    raise RuntimeError(
      f'Expected {nao_sup} fragment MOs, got {C_all.shape[1]}. '
      'The fragment basis/AO ordering does not match the supersystem.'
    )

  allowed = np.array([0.0, 1.0, 2.0])
  if not np.all(np.any(np.isclose(occ_all[:, None], allowed[None, :]), axis=1)):
    raise RuntimeError('Unexpected fragment MO occupations; expected 0, 1 or 2.')

  S = mf_sup.get_ovlp()

  C2 = C_all[:, np.isclose(occ_all, 2.0)]
  C1 = C_all[:, np.isclose(occ_all, 1.0)]
  C0 = C_all[:, np.isclose(occ_all, 0.0)]

  Q2 = metric_orthogonalize(C2, S)
  Q1 = metric_orthogonalize(C1, S, against=Q2)
  Qocc = np.hstack((Q2, Q1))
  Q0 = metric_orthogonalize(C0, S, against=Qocc)

  C_sup = np.hstack((Q2, Q1, Q0))
  occ_sup = np.concatenate((
    np.full(Q2.shape[1], 2.0),
    np.full(Q1.shape[1], 1.0),
    np.zeros(Q0.shape[1])
  ))

  metric_error = np.max(np.abs(C_sup.conj().T @ S @ C_sup - np.eye(nao_sup)))
  if metric_error > 1e-7:
    raise RuntimeError(f'S-orthonormality check failed: {metric_error:.3e}')

  nalpha = np.count_nonzero(occ_sup > 0)
  nbeta = np.count_nonzero(occ_sup > 1)
  if nalpha != (mol_sup.nelectron + mol_sup.spin) // 2:
    raise RuntimeError(
      f'Alpha electron count mismatch: {nalpha} vs '
      f'{(mol_sup.nelectron + mol_sup.spin)//2}'
    )
  if nbeta != (mol_sup.nelectron - mol_sup.spin) // 2:
    raise RuntimeError(
      f'Beta electron count mismatch: {nbeta} vs '
      f'{(mol_sup.nelectron - mol_sup.spin)//2}'
    )

  F = mf_sup.get_fock(dm=dm_sup)
  eps = np.einsum('pi,pq,qi->i', C_sup.conj(), F, C_sup).real

  mf_sup.mo_coeff = C_sup
  mf_sup.mo_occ = occ_sup
  mf_sup.mo_energy = eps

  #print('Fragment-derived supersystem orbitals:')
  #print('  n(MO) =', C_sup.shape[1])
  #print('  n(double/single/virtual) =', Q2.shape[1], Q1.shape[1], Q0.shape[1])
  #print('  alpha/beta electrons =', nalpha, nbeta)
  #print('  max |C^dagger S C - I| =', metric_error)

  return mf_sup


def supsystem_hf(work_dir, mol_sup, atoms, D_a_init, D_a_init_sym, mol_a, mol_b, re_element, C_a, occ_a, C_a_sym, occ_a_sym):

  #print (''.join(('Step_3 Super System Calculation')).center(80,'*'), flush=True)
 
  if mol_a.spin == 0:
    mf = scf.RHF(mol_sup).x2c()
    mf.level_shift = 3.0
    #dm_tot = dm_sup[0] + dm_sup[1]
    ligand_ene = mf.kernel()
  else:
    nao_sup = mol_sup.nao_nr()
    sup_slices = basis_info.get_sup_ao_indices(mol_sup)
    a_atom_indices, b_atom_indices = read_coord.get_atom_indices(atoms)
    a_ao_indices = basis_info.get_atom_ao_indices(a_atom_indices, sup_slices)
    b_ao_indices = basis_info.get_atom_ao_indices(b_atom_indices, sup_slices)
 
    norb, ne_act, mo = cahf_init_guess.atom_init(mol_sup, re_element)
    #ne_act = 3
    if ( not ne_act == re_info.get_f_elec_num(re_element) ):
      ligand_ene, D_b_init, C_b, occ_b, eps_b = subsystem.subsystem_b_hf(work_dir, mol_b, mol_a)
      norb, ne_act, mo = cahf_init_guess.define_occ_dm(mol_sup, nao_sup, a_ao_indices, b_ao_indices, D_a_init_sym, D_b_init, re_element)
    if ( not ne_act == re_info.get_f_elec_num(re_element) ):
      norb, ne_act, mo = cahf_init_guess.define_frac_orb(mol_sup, nao_sup, C_a_sym, occ_a_sym, C_b, occ_b, \
                                                         a_ao_indices, b_ao_indices, D_a_init_sym, D_b_init, re_element)
    if ( not ne_act == re_info.get_f_elec_num(re_element) ):
      norb, ne_act, mo = cahf_init_guess.define_frac_orb(mol_sup, nao_sup, C_a, occ_a, C_b, occ_b, a_ao_indices, \
                                                         b_ao_indices, D_a_init, D_b_init, re_element)

    if ( ne_act == re_info.get_f_elec_num(re_element) ):
      print ('Success: the f electron number of %s is %d for CAHF initial guess' %(re_element, ne_act), flush=True)
    else:
      print ('Failure: the f electron number for  CAHF initial guess should be %d, not %d' \
             %(ne_act, re_info.get_f_elec_num(re_element)), flush=True)
      exit()

    # Retain the active-space choice used in the original script.
    mf = scf.rohf.ROHF(mol_sup).x2c()
    mf.max_cycle = 0
    mf.kernel()

    ncas = 7
    nelec = re_info.get_f_elec_num(re_element)
    occ = cahf.CAHF_get_occ(ncas, nelec)(mf)

    if ( nao_sup > 1000 ):
      mydf = df.DF(mol_sup)
      mydf.auxbasis = df.make_auxbasis(mol_sup)
      h5_file = ''.join((work_dir, '/', str(uuid.uuid4()), '.h5'))
      mydf._cderi_to_save = h5_file
      mydf.build()

      mf_cahf = cahf.CAHF(
      mol_sup, ncas=ncas, nelecas=nelec, spin=re_info.get_spin(re_element)
      ).x2c().newton().density_fit()
      mf_cahf.with_df._cderi = h5_file

    else:
      mf_cahf = cahf.CAHF(
      mol_sup, ncas=ncas, nelecas=nelec, spin=re_info.get_spin(re_element)
      ).x2c().newton()

    ligand_ene = mf_cahf.kernel(mo, occ)
    mf = mf_cahf

    if ( nao_sup > 1000 ):
      subprocess.run('rm %s' %(h5_file), cwd=work_dir, shell=True)

  dm = mf.make_rdm1()

  ao_labels = ['%s 4f' % re_element]
  norb, ne_act, orbs = myavas.avas(mf, ao_labels, threshold=0.5)
  if ( ne_act == re_info.get_f_elec_num(re_element) ):
    print ('Success: the f electron number of %s is %d in CAHF calculation' %(re_element, ne_act), flush=True)
  else:
    print ('Failure: the f electron number should be %d, but CAHF result is %d' \
          %(re_info.get_f_elec_num(re_element), ne_act), flush=True)

  print ('The final CAHF energy is %f Hartree' %(ligand_ene))
  if mol_a.spin == 0:
    dm_a = dm / 2.0
    dm_b = dm / 2.0
  else:
    dm_a = dm[0]
    dm_b = dm[1]

  return ligand_ene, [dm_a, dm_b], mf.mo_coeff

def supersystem_hf_mm(mol, mm_hcore_file, mo, re_element):

  mf = cahf.CAHF(mol, ncas=7, nelecas=re_info.get_f_elec_num(re_element), spin=re_info.get_spin(re_element)).x2c()
  mf.init_guess = 'atom'
  mf.max_cycle=0
  mf.kernel()
  occ = cahf.CAHF_get_occ(7, re_info.get_f_elec_num(re_element))(mf)

  #mf = cahf.CAHF(mol, ncas=7, nelecas=re_info.get_f_elec_num(re_element), spin=re_info.get_spin(re_element)).x2c().newton()
  mf = cahf.CAHF(mol, ncas=7, nelecas=re_info.get_f_elec_num(re_element), spin=re_info.get_spin(re_element)).x2c()
  get_hcore_original = mf.get_hcore
  hcore_mm = np.load(mm_hcore_file)
  def get_hcore_mm(self, mol=None):
    return get_hcore_original(mol=mol) + hcore_mm
  import types
  mf.get_hcore = types.MethodType(get_hcore_mm, mf)
  mf.max_cycle=200
  mf.conv_tol = 1e-9
  #dm = mf.make_rdm1(mo, occ)
  #qmmm_ene = mf.kernel(dm)

  mf = mf.newton()
  qmmm_ene = mf.kernel(mo, occ)

  #print(mf._scf.get_hcore(mol))

  ao_labels = ['%s 4f' % re_element]
  norb, ne_act, orbs = myavas.avas(mf, ao_labels, threshold=0.5)
  if ( ne_act == re_info.get_f_elec_num(re_element) ):
    print ('Success: the f electron number of %s is %d in QM/MM calculation' %(re_element, ne_act), flush=True)
  else:
    print ('Failure: the f electron number should be %d, but QM/MM result is %d' \
          %(re_info.get_f_elec_num(re_element), ne_act), flush=True)

  print ('The final QM/MM energy is %f Hartree' %(qmmm_ene))

  return mf


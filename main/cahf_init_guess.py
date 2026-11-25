#! /usr/env/bin python

from pyscf import scf
from EMBRE_SS.main import myavas
from EMBRE_SS.main import supersystem

def atom_init(mol_sup, re_element):

  mf = scf.rohf.ROHF(mol_sup).x2c()
  mf.init_guess = 'atom'
  mf.max_cycle=0
  mf.kernel()
  norb, ne_act, mo = myavas.avas(mf, ['%s 4f' % re_element], threshold=0.5)

  return norb, ne_act, mo

def define_occ_dm(mol_sup, nao_sup, a_ao_indices, b_ao_indices, D_a, D_b, re_element):

  mf = scf.rohf.ROHF(mol_sup).x2c()
  D_sup_alpha, D_sup_beta = supersystem.make_sup_density(nao_sup, a_ao_indices, b_ao_indices, D_a, D_b)
  mf.max_cycle=0
  mf.kernel([D_sup_alpha, D_sup_beta])
  norb, ne_act, mo =  myavas.avas(mf, ['%s 4f' % re_element], threshold=0.5)

  return norb, ne_act, mo

def define_frac_orb(mol_sup, nao_sup, C_a, occ_a, C_b, occ_b, a_ao_indices, b_ao_indices, D_a, D_b, re_element):

  mf = scf.rohf.ROHF(mol_sup).x2c()
  D_sup_alpha, D_sup_beta = supersystem.make_sup_density(nao_sup, a_ao_indices, b_ao_indices, D_a, D_b)
  mf = supersystem.build_fragment_mo_set(mol_sup, mf, C_a, occ_a, C_b, occ_b, a_ao_indices, b_ao_indices, [D_sup_alpha, D_sup_beta])
  norb, ne_act, mo =  myavas.avas(mf, ['%s 4f' % re_element], threshold=0.5)

  return norb, ne_act, mo

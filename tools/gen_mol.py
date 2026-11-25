#! /usr/env/bin python

from pyscf import gto
from pyscf.pbc import gto as pbcgto
from EMBRE_SS.tools import re_info
from EMBRE_SS.tools import read_coord
from EMBRE_SS.data import ln_basis

def get_sub_a_mol(re_atoms):

  re_element = read_coord.get_re_element(re_atoms[0])
  basis = ln_basis.get_basis(re_element)
  spin = re_info.get_spin(re_element)

  mol = gto.Mole()
  mol.atom = re_atoms
  mol.basis = {re_element: gto.parse(basis)}
  mol.charge = +3
  mol.verbose = 0
  mol.spin = spin
  #mol.symmetry = False
  mol.build()

  return mol

def get_sub_b_mol(ligand_atoms, charge):

  mol = gto.Mole()
  mol.atom = ligand_atoms
  natm = len(ligand_atoms)
  if ( natm > 10 ):
    mol.basis = 'cc-pVDZ'
  else:
    mol.basis = 'cc-pVTZ'
  mol.charge = charge
  mol.verbose = 0
  mol.spin = 0
  mol.build()

  return mol

def get_sup_mol(mol_a, mol_b, qm_atoms):

  atom_ab = []

  for i in range(mol_a.natm):
    atom_ab.append([mol_a.atom_symbol(i), mol_a.atom_coord(i)])

  for i in range(mol_b.natm):
    atom_ab.append([mol_b.atom_symbol(i), mol_b.atom_coord(i)])

  basis_ab = {}

  for i in range(mol_a.natm):
    label = mol_a.atom_symbol(i)
    basis_ab[label] = mol_a._basis[label]

  for i in range(mol_b.natm):
    label = mol_b.atom_symbol(i)
    basis_ab[label] = mol_b._basis[label]

  mol_sup = gto.Mole()

  #mol_sup.atom = atom_ab
  mol_sup.atom = qm_atoms
  mol_sup.basis = basis_ab

  mol_sup.charge = mol_a.charge + mol_b.charge
  mol_sup.spin = mol_a.spin + mol_b.spin
  mol_sup.verbose = 0

  mol_sup.build()

  return mol_sup

def get_cell(mol, cell_vector):

  cell = pbcgto.Cell()
  cell.spin = mol.spin
  cell.charge = mol.charge
  cell.basis = mol.basis
  cell.atom = mol.atom
  cell.a = cell_vector
  cell.build()

  return cell

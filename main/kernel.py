#! /usr/env/bin python

import sys
import numpy as np
from EMBRE_SS.tools import read_coord
from EMBRE_SS.tools import gen_mol
from EMBRE_SS.tools import basis_info
from EMBRE_SS.tools import get_qm_mm
from EMBRE_SS.main import subsystem 
from EMBRE_SS.main import supersystem
from EMBRE_SS.main import get_mm_hcore
from EMBRE_SS.main import casscf_run
from EMBRE_SS.main import nevpt2_run
from EMBRE_SS.main import soc

work_dir = sys.argv[1]
coord_file = sys.argv[2]
cell_file = sys.argv[3]
charge_file = sys.argv[4]

cell_vector = read_coord.get_box(cell_file)
elements, xyz = read_coord.read_last_frame(coord_file)
atoms = [[elem, np.array(coord, dtype=float)] for elem, coord in zip(elements, xyz)]
qm_indices, mm_indices = get_qm_mm.get_qm_mm_index(elements, xyz, cell_vector)
charges = read_coord.get_bader_charge(charge_file, coord_file)

charge = -6

qm_atoms = [atoms[i] for i in qm_indices]
re_atoms, ligand_atoms, re_element = read_coord.split_re_ligand(qm_atoms)

mol_a = gen_mol.get_sub_a_mol(re_atoms)
mol_b = gen_mol.get_sub_b_mol(ligand_atoms, charge-3)
mol_sup = gen_mol.get_sup_mol(mol_a, mol_b, qm_atoms)

print (''.join(('Step 1')).center(80,'*'), flush=True)
print ('Calculate the electrostatic potential for the QM region', flush=True)
mm_hcore_file = get_mm_hcore.mm_hcore(mol_sup, atoms, charges, cell_vector, qm_indices, mm_indices, work_dir)

print (mm_hcore_file)

re_ene, D_a_init, C_a, occ_a, eps_a = subsystem.subsystem_a_hf(mol_a, re_element)
re_ene_sym, D_a_init_sym, C_a_sym, occ_a_sym, eps_a_sym = subsystem.subsystem_a_hf_sym(mol_a, re_element)

print (''.join(('Step 2')).center(80,'*'), flush=True)
print ('Perform the SCF calculation of the QM region', flush=True)
re_ligand_ene, D_sup, mo = supersystem.supsystem_hf(work_dir, mol_sup, qm_atoms, D_a_init, D_a_init_sym, mol_a, \
                                                    mol_b, re_element, C_a, occ_a, C_a_sym, occ_a_sym)

print (''.join(('Step 3')).center(80,'*'), flush=True)
print ('Perform the SCF calculation of the QM region with the electrostatic potential', flush=True)
mf = supersystem.supersystem_hf_mm(mol_sup, mm_hcore_file, mo, re_element)

print (''.join(('Step 4')).center(80,'*'), flush=True)
print ('Perform the CASSCF calculation', flush=True)
solver, statelis = casscf_run.kernel(mf, mol_sup, re_element)

print (''.join(('Step 5')).center(80,'*'), flush=True)
print ('Perform the NEVPT2 calculation for each state', flush=True)
solver = nevpt2_run.kernel(solver, statelis, re_element)

print (''.join(('Step 6')).center(80,'*'), flush=True)
print ('Perform the CASSCF/NEVPT2/SO calculation', flush=True)
mysiso = soc.SISO('debug_all', solver, 'bp', statelis)
mysiso.kernel()
mysiso.analyze()


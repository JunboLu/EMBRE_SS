#! /usr/env/bin python

import uuid
import subprocess
from pathlib import Path
from pyscf import df
from pyscf import scf
from EMBRE_SS.main import myavas
from EMBRE_SS.tools import re_info

LIB_DIR = Path(__file__).resolve().parent.parent / 'lib'

def subsystem_a_hf_sym(mol, re_element):

  mol_a = mol.copy()
  mol_a.symmetry = True
  mol_a.build()

  #print (''.join(('Step_1 Ln3+_Calculation')).center(80,'*'), flush=True)

  if ( mol_a.spin == 0 ):
    mf = scf.RHF(mol_a)
    mf.level_shift = 2.0
    re_ene = mf.kernel()
    dm = mf.make_rdm1()
    dm_a = dm/2.0
    dm_b = dm/2.0
  else:
    mf = scf.UHF(mol_a).x2c()
    if ( re_element == 'Ce' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (0,0), 'f+1': (0,0), 'f+2': (0,0), 'f+3': (1,0), 'f-1': (0,0), 'f-2': (0,0), 'f-3': (0,0)}
    if ( re_element == 'Pr' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (0,0), 'f+1': (0,0), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (0,0), 'f-2': (0,0), 'f-3': (0,0)}
    if ( re_element == 'Nd' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (0,0), 'f+1': (1,0), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (0,0), 'f-2': (0,0), 'f-3': (0,0)}
    if ( re_element == 'Pm' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (1,0), 'f+1': (1,0), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (0,0), 'f-2': (0,0), 'f-3': (0,0)}
    if ( re_element == 'Sm' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (1,0), 'f+1': (1,0), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (1,0), 'f-2': (0,0), 'f-3': (0,0)}
    if ( re_element == 'Eu' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (1,0), 'f+1': (1,0), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (1,0), 'f-2': (1,0), 'f-3': (0,0)}
    if ( re_element == 'Gd' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (1,0), 'f+1': (1,0), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (1,0), 'f-2': (1,0), 'f-3': (1,0)}
    if ( re_element == 'Tb' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (1,0), 'f+1': (1,0), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (1,0), 'f-2': (1,0), 'f-3': (1,1)}
    if ( re_element == 'Dy' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (1,0), 'f+1': (1,0), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (1,0), 'f-2': (1,1), 'f-3': (1,1)}
    if ( re_element == 'Ho' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (1,0), 'f+1': (1,0), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (1,1), 'f-2': (1,1), 'f-3': (1,1)}
    if ( re_element == 'Er' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (1,1), 'f+1': (1,0), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (1,1), 'f-2': (1,1), 'f-3': (1,1)}
    if ( re_element == 'Tm' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (1,1), 'f+1': (1,1), 'f+2': (1,0), 'f+3': (1,0), 'f-1': (1,1), 'f-2': (1,1), 'f-3': (1,1)}
    if ( re_element == 'Yb' ):
      mf.irrep_nelec = {'s+0': (5,5), 'p+0': (4,4), 'p-1': (4,4), 'p+1': (4,4), 'd+0': (2,2), 'd-1': (2,2), 'd-2': (2,2), 'd+1': (2,2), 'd+2': (2,2), 'f+0': (1,1), 'f+1': (1,1), 'f+2': (1,1), 'f+3': (1,0), 'f-1': (1,1), 'f-2': (1,1), 'f-3': (1,1)}
    mf.level_shift = 3.0
    mf.kernel()

    if not mf.converged:
      mf = mf.newton()
      mf.kernel()

    mol_a.symmetry = False
    mol_a.build()

    dm = mf.make_rdm1()
    mf = scf.rohf.ROHF(mol_a).x2c()
    mf.max_cycle = 0
    re_ene = mf.kernel(dm)

    ao_labels = ['%s 4f'%(re_element)]
    norb, ne_act, orbs = myavas.avas(mf, ao_labels, threshold=0.5)
    #if ( ne_act == re_info.get_f_elec_num(re_element) ):
    #  print ('Success: the f electron number of %s is %d' %(re_element, ne_act))

    dm = mf.make_rdm1()
    dm_a = dm[0]
    dm_b = dm[1]

    mo_occ = mf.mo_occ.copy()
    mo_coeff = mf.mo_coeff.copy()
    mo_ene = mf.mo_energy.copy()

    #print ('The energy of Ln3+ is %f Hartree' %(re_ene))

  return re_ene, [dm_a, dm_b], mo_coeff, mo_occ, mo_ene

def subsystem_a_hf(mol_a, re_element):

  #print (''.join(('Step_1 Ln3+_Calculation')).center(80,'*'), flush=True)

  if ( mol_a.spin == 0 ):
    mf = scf.RHF(mol_a)
    mf.level_shift = 2.0
    re_ene = mf.kernel()
    dm = mf.make_rdm1()
    dm_a = dm/2.0
    dm_b = dm/2.0
  else:
    mol_a.spin = 0
    mf = scf.RHF(mol_a).x2c()
    dm = mf.from_chk(''.join((str(LIB_DIR), '/wfn/', re_element, '.chk')))
    mf.max_cycle = 100
    mf.level_shift = 2.0
    mf = scf.addons.frac_occ(mf)
    re_ene = mf.kernel(dm)

    mol_a.spin = re_info.get_spin(re_element)

    ao_labels = ['%s 4f'%(re_element)]
    norb, ne_act, orbs = myavas.avas(mf, ao_labels, threshold=0.5)
    #if ( ne_act == re_info.get_f_elec_num(re_element) ): 
    #  print ('Success: the f electron number of %s is %d' %(re_element, ne_act))
    dm = mf.make_rdm1()
    dm_a = dm/2.0
    dm_b = dm/2.0

    double_occ_num = int((re_info.get_tot_elec_num(re_element)-re_info.get_f_elec_num(re_element))/2) + \
                     int((re_info.get_f_elec_num(re_element)-re_info.get_spin(re_element))/2)
    singly_occ_num = re_info.get_spin(re_element)
    vir_occ_num = mol_a.nao_nr() - double_occ_num - singly_occ_num
    mo_occ = [2.0]*double_occ_num + [1.0]*singly_occ_num + [0.0]*vir_occ_num
    mo_coeff = mf.mo_coeff.copy()
    mo_ene = mf.mo_energy.copy()

    #print ('The energy of Ln3+ is %f Hartree' %(re_ene))

  return re_ene, [dm_a, dm_b], mo_coeff, mo_occ, mo_ene

def subsystem_b_hf(work_dir, mol_b, mol_a):

  #print (''.join(('Step_2 Ligand_Calculation')).center(80,'*'), flush=True)

  if ( mol_a.spin == 0 ):
    if ( mol_b.nao_nr() > 500 ):
      mydf = df.DF(mol_b)
      mydf.auxbasis = df.make_auxbasis(mol_b)
      h5_file = ''.join((work_dir, '/', str(uuid.uuid4()), '.h5'))
      mydf._cderi_to_save = h5_file
      mydf.build()

      mf = scf.RHF(mol_b).x2c().density_fit()
      mf.with_df._cderi = h5_file
    else:
      mf = scf.RHF(mol_b).x2c()
  else:
    if ( mol_b.nao_nr() > 500 ):
      mydf = df.DF(mol_b)
      mydf.auxbasis = df.make_auxbasis(mol_b)
      h5_file = ''.join((work_dir, '/', str(uuid.uuid4()), '.h5'))
      mydf._cderi_to_save = h5_file
      mydf.build()

      mf = scf.ROHF(mol_b).x2c().density_fit()
      mf.with_df._cderi = h5_file
    else:
      mf = scf.ROHF(mol_b).x2c()

  mf.level_shift = 2.0
  ligand_ene = mf.kernel()
  dm = mf.make_rdm1()

  if not mf.converged:
    mf = mf.newton()
    ligand_ene = mf.kernel()
    dm = mf.make_rdm1()
    if not mf.converged:
      print ('Subsystem b does not converge!')
      exit()

  if ( mol_b.nao_nr() > 1000 ):
    subprocess.run('rm %s' %(h5_file), cwd=work_dir, shell=True)

  #print ('Success: The SCF calculation of ligand is converged')

  if ( mol_a.spin == 0 ):
    dm_a = dm/2.0
    dm_b = dm/2.0
  else:
    dm_a = dm[0]
    dm_b = dm[1]

  mo_coeff = mf.mo_coeff.copy()
  mo_occ = mf.mo_occ.copy()
  mo_ene = mf.mo_energy.copy()

  #print ('The energy of ligand is %f Hartree' %(ligand_ene))

  return ligand_ene, [dm_a, dm_b], mo_coeff, mo_occ, mo_ene


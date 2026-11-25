#!/usr/bin/env python

import numpy as np
from pyscf import mcscf
from pyscf import fci
from EMBRE_SS.tools import re_info

def Ce_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 1
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=0.5)
  solver1.nroots = 7
  solvers.append(solver1)

  weights = np.ones(solver1.nroots)/(solver1.nroots)
  statelis = [0, 7]

  return solvers, weights, statelis

def Pr_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 0
  solver1 = fci.addons.fix_spin(solver1,ss=0,shift=.5)
  solver1.nroots = 28
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 2
  solver2 = fci.addons.fix_spin(solver2,ss=2,shift=.5)
  solver2.nroots = 21
  solvers.append(solver2)

  weights = np.ones(solver1.nroots+solver2.nroots)/(solver1.nroots+solver2.nroots)
  statelis = [28, 0, 21, 0]

  return solvers, weights, statelis

def Nd_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 1
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=.5)
  solver1.nroots = 112
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 3
  solver2 = fci.addons.fix_spin(solver2,ss=(float(solver2.spin)/2.0)*(float(solver2.spin)/2.0+1.0),shift=.5)
  solver2.nroots = 35
  solvers.append(solver2)

  weights = np.ones(solver1.nroots+solver2.nroots)/(solver1.nroots+solver2.nroots)
  statelis = [0, 112, 0, 35]

  return solvers, weights, statelis

def Pm_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 0
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=2.0)
  solver1.nroots = 96
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 2
  solver2 = fci.addons.fix_spin(solver2,ss=(float(solver2.spin)/2.0)*(float(solver2.spin)/2.0+1.0),shift=2.0)
  solver2.nroots = 126
  solvers.append(solver2)

  solver3 = fci.direct_spin1.FCI(mol)
  solver3.spin = 4
  solver3 = fci.addons.fix_spin(solver3,ss=(float(solver3.spin)/2.0)*(float(solver3.spin)/2.0+1.0),shift=2.0)
  solver3.nroots = 35
  solvers.append(solver3)

  weights = np.ones(solver1.nroots+solver2.nroots+solver3.nroots)/(solver1.nroots+solver2.nroots+solver3.nroots)
  statelis = [96, 0, 126, 0, 35, 0]

  return solvers, weights, statelis

def Sm_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 1
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=1.0)
  solver1.nroots = 158
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 3
  solver2 = fci.addons.fix_spin(solver2,ss=(float(solver2.spin)/2.0)*(float(solver2.spin)/2.0+1.0),shift=1.0)
  solver2.nroots = 128
  solvers.append(solver2)

  solver3 = fci.direct_spin1.FCI(mol)
  solver3.spin = 5
  solver3 = fci.addons.fix_spin(solver3,ss=(float(solver3.spin)/2.0)*(float(solver3.spin)/2.0+1.0),shift=1.0)
  solver3.nroots = 21
  solvers.append(solver3)

  weights = np.ones(solver1.nroots+solver2.nroots+solver3.nroots)/(solver1.nroots+solver2.nroots+solver3.nroots)
  statelis = [0, 158, 0, 128, 0, 21]

  return solvers, weights, statelis

def Eu_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 0
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=1.0)
  solver1.nroots = 86
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 2
  solver2 = fci.addons.fix_spin(solver2,ss=(float(solver2.spin)/2.0)*(float(solver2.spin)/2.0+1.0),shift=1.0)
  solver2.nroots = 112
  solvers.append(solver2)

  solver3 = fci.direct_spin1.FCI(mol)
  solver3.spin = 4
  solver3 = fci.addons.fix_spin(solver3,ss=(float(solver3.spin)/2.0)*(float(solver3.spin)/2.0+1.0),shift=1.0)
  solver3.nroots = 86
  solvers.append(solver3)

  solver4 = fci.direct_spin1.FCI(mol)
  solver4.spin = 6
  solver4 = fci.addons.fix_spin(solver4,ss=(float(solver4.spin)/2.0)*(float(solver4.spin)/2.0+1.0),shift=1.0)
  solver4.nroots = 7
  solvers.append(solver4)

  weights = np.ones(solver1.nroots+solver2.nroots+solver3.nroots+solver4.nroots)/(solver1.nroots+solver2.nroots+solver3.nroots+solver4.nroots)
  statelis = [86, 0, 112, 0, 86, 0, 7, 0]

  return solvers, weights, statelis

def Gd_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 1
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=1.0)
  solver1.nroots = 134
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 3
  solver2 = fci.addons.fix_spin(solver2,ss=(float(solver2.spin)/2.0)*(float(solver2.spin)/2.0+1.0),shift=1.0)
  solver2.nroots = 156
  solvers.append(solver2)

  solver3 = fci.direct_spin1.FCI(mol)
  solver3.spin = 5
  solver3 = fci.addons.fix_spin(solver3,ss=(float(solver3.spin)/2.0)*(float(solver3.spin)/2.0+1.0),shift=1.0)
  solver3.nroots = 48
  solvers.append(solver3)

  solver4 = fci.direct_spin1.FCI(mol)
  solver4.spin = 7
  solver4 = fci.addons.fix_spin(solver4,ss=(float(solver4.spin)/2.0)*(float(solver4.spin)/2.0+1.0),shift=1.0)
  solver4.nroots = 1
  solvers.append(solver4)

  weights = np.ones(solver1.nroots+solver2.nroots+solver3.nroots+solver4.nroots)/(solver1.nroots+solver2.nroots+solver3.nroots+solver4.nroots)
  statelis = [0, 134, 0, 156, 0, 48, 0, 1]

  return solvers, weights, statelis

def Tb_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 0
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=1.0)
  solver1.nroots = 86
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 2
  solver2 = fci.addons.fix_spin(solver2,ss=(float(solver2.spin)/2.0)*(float(solver2.spin)/2.0+1.0),shift=1.0)
  solver2.nroots = 112
  solvers.append(solver2)

  solver3 = fci.direct_spin1.FCI(mol)
  solver3.spin = 4
  solver3 = fci.addons.fix_spin(solver3,ss=(float(solver3.spin)/2.0)*(float(solver3.spin)/2.0+1.0),shift=1.0)
  solver3.nroots = 86
  solvers.append(solver3)

  solver4 = fci.direct_spin1.FCI(mol)
  solver4.spin = 6
  solver4 = fci.addons.fix_spin(solver4,ss=(float(solver4.spin)/2.0)*(float(solver4.spin)/2.0+1.0),shift=1.0)
  solver4.nroots = 7
  solvers.append(solver4)

  weights = np.ones(solver1.nroots+solver2.nroots+solver3.nroots+solver4.nroots)/(solver1.nroots+solver2.nroots+solver3.nroots+solver4.nroots)
  statelis = [86, 0, 112, 0, 86, 0, 7, 0]

  return solvers, weights, statelis

def Dy_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 1
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=1.0)
  solver1.nroots = 158
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 3
  solver2 = fci.addons.fix_spin(solver2,ss=(float(solver2.spin)/2.0)*(float(solver2.spin)/2.0+1.0),shift=1.0)
  solver2.nroots = 128
  solvers.append(solver2)

  solver3 = fci.direct_spin1.FCI(mol)
  solver3.spin = 5
  solver3 = fci.addons.fix_spin(solver3,ss=(float(solver3.spin)/2.0)*(float(solver3.spin)/2.0+1.0),shift=1.0)
  solver3.nroots = 21
  solvers.append(solver3)

  weights = np.ones(solver1.nroots+solver2.nroots+solver3.nroots)/(solver1.nroots+solver2.nroots+solver3.nroots)
  statelis = [0, 158, 0, 128, 0, 21]

  return solvers, weights, statelis

def Ho_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 0
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=1.0)
  solver1.nroots = 96
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 2
  solver2 = fci.addons.fix_spin(solver2,ss=(float(solver2.spin)/2.0)*(float(solver2.spin)/2.0+1.0),shift=1.0)
  solver2.nroots = 126
  solvers.append(solver2)

  solver3 = fci.direct_spin1.FCI(mol)
  solver3.spin = 4
  solver3 = fci.addons.fix_spin(solver3,ss=(float(solver3.spin)/2.0)*(float(solver3.spin)/2.0+1.0),shift=1.0)
  solver3.nroots = 35
  solvers.append(solver3)

  weights = np.ones(solver1.nroots+solver2.nroots+solver3.nroots)/(solver1.nroots+solver2.nroots+solver3.nroots)
  statelis = [96, 0, 126, 0, 35]

  return solvers, weights, statelis

def Er_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 1
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=1.0)
  solver1.nroots = 112
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 3
  solver2 = fci.addons.fix_spin(solver2,ss=(float(solver2.spin)/2.0)*(float(solver2.spin)/2.0+1.0),shift=1.0)
  solver2.nroots = 35
  solvers.append(solver2)

  weights = np.ones(solver1.nroots+solver2.nroots)/(solver1.nroots+solver2.nroots)
  statelis = [0, 112, 0, 35]

  return solvers, weights, statelis

def Tm_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 0
  solver1 = fci.addons.fix_spin(solver1,ss=0,shift=1.0)
  solver1.nroots = 28
  solvers.append(solver1)

  solver2 = fci.direct_spin1.FCI(mol)
  solver2.spin = 2
  solver2 = fci.addons.fix_spin(solver2,ss=2,shift=1.0)
  solver2.nroots = 21
  solvers.append(solver2)

  weights = np.ones(solver1.nroots+solver2.nroots)/(solver1.nroots+solver2.nroots)
  statelis = [28, 0, 21, 0]

  return solvers, weights, statelis

def Yb_solver(mol):

  solvers = []
  solver1 = fci.direct_spin1.FCI(mol)
  solver1.spin = 1
  solver1 = fci.addons.fix_spin(solver1,ss=(float(solver1.spin)/2.0)*(float(solver1.spin)/2.0+1.0),shift=1.0)
  solver1.nroots = 7
  solvers.append(solver1)

  weights = np.ones(solver1.nroots)/(solver1.nroots)
  statelis = [0, 7]

  return solvers, weights, statelis

def kernel(mf, mol, re_element):

  solver = mcscf.CASSCF(mf, 7, re_info.get_f_elec_num(re_element))

  if ( re_element == 'Ce' ):
    solvers, weights, statelis = Ce_solver(mol)
  if ( re_element == 'Pr' ):
    solvers, weights, statelis = Pr_solver(mol)
  if ( re_element == 'Nd' ):
    solvers, weights, statelis = Nd_solver(mol)
  if ( re_element == 'Pm' ):
    solvers, weights, statelis = Pm_solver(mol)
  if ( re_element == 'Sm' ):
    solvers, weights, statelis = Sm_solver(mol)
  if ( re_element == 'Eu' ):
    solvers, weights, statelis = Eu_solver(mol)
  if ( re_element == 'Gd' ):
    solvers, weights, statelis = Gd_solver(mol)
  if ( re_element == 'Tb' ):
    solvers, weights, statelis = Tb_solver(mol)
  if ( re_element == 'Dy' ):
    solvers, weights, statelis = Dy_solver(mol)
  if ( re_element == 'Ho' ):
    solvers, weights, statelis = Ho_solver(mol)
  if ( re_element == 'Er' ):
    solvers, weights, statelis = Er_solver(mol)
  if ( re_element == 'Tm' ):
    solvers, weights, statelis = Tm_solver(mol)
  if ( re_element == 'Yb' ):
    solvers, weights, statelis = Yb_solver(mol)

  mcscf.state_average_mix_(solver, solvers, weights)
  solver.run(mf.mo_coeff)

  print ('The final CASSCF energy is %f Hartree' %(solver.e_tot))

  return solver, statelis


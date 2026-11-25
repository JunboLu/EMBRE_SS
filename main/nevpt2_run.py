#! /usr/env/bin/python

from pyscf import mcscf
from pyscf import fci
from pyscf import mrpt
from EMBRE_SS.tools import re_info

def kernel(solver, statelis, re_element):

  e_corr = []
  mo = solver.mo_coeff
  ncas = 7
  nelecas = re_info.get_f_elec_num(re_element) 

  for i in range(len(statelis)):
    e_corr_i = []
    if statelis[i]:
      nelecas = fci.addons._unpack_nelec(nelecas, i)
      mc_ci = mcscf.CASCI(mf, ncas, nelecas)
      #print('CASCI fixed_spin_', (i/2)*(i/2+1))
      mc_ci.fcisolver.spin = i
      fix_spin_shift = 0.5
      mc_ci.fix_spin_(shift=fix_spin_shift, ss=(i/2)*(i/2+1))
      mc_ci.fcisolver.nroots = statelis[i]
      mc_ci.kernel(mo)
      #print('mc_ci.nelecas', mc_ci.nelecas)
      mc_ci.nelecas = fci.addons._unpack_nelec(nelecas, i)
      #print('mc_ci.nelecas', mc_ci.nelecas)
      for iroot in range(statelis[i]):
        e_corr_i.append(mrpt.NEVPT(mc_ci, root=iroot).kernel())

    e_corr = e_corr+e_corr_i

  for i in range(len(solver.e_states)):
    solver.e_states[i] = solver.e_states[i] + e_corr[i]

  return solver

#! /usr/env/bin/python

import numpy as np
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from pyscf import gto, lib

def first_shell(els, xyz, cell, center="auto", target="F", cut=None, dr=0.05):

  cell = np.asarray(cell, float)
  vol = abs(np.linalg.det(cell))
  if center == "auto":
      rare = sorted(set(els) - {"La", "F"})
      center = rare[0] if rare else min(set(els), key=els.count)
  if center not in els:
      raise ValueError('元素列表里没有中心原子 %s（现有元素: %s）'
                       % (center, ', '.join(sorted(set(els)))))
  ci = els.index(center) 
  ti = [i for i, e in enumerate(els) if e == target and i != ci]
  if not ti:
      raise ValueError('团簇里没有配位原子 %s（现有元素: %s）'
                       % (target, ', '.join(sorted(set(els)))))

  shifts = [i*cell[0] + j*cell[1] + k*cell[2] for i, j, k in product((-1, 0, 1), repeat=3)]
  d = np.min([np.linalg.norm(xyz[ci] - xyz[ti] + s, axis=1) for s in shifts], axis=0)

  if cut is None:                       # 自动找第一配位壳的边界
      spac = [vol / np.linalg.norm(np.cross(cell[(i+1) % 3], cell[(i+2) % 3]))
              for i in range(3)]
      edges = np.arange(0.0, min(0.5*min(spac), 8.0) + dr, dr)
      r = 0.5 * (edges[1:] + edges[:-1])
      g = np.histogram(d, bins=edges)[0] / (4 * np.pi * r**2 * dr * len(ti) / vol)
      i0 = int(np.searchsorted(r, 1.5))
      ip = i0 + int(np.argmax(g[i0:]))
      thr, i, cut = 0.1 * g[ip], ip, r[-1]
      while i < len(r):
          while i < len(r) and g[i] >= thr:
              i += 1
          j = i
          while j < len(r) and g[j] < thr:
              j += 1
          if j >= len(r) or r[min(j, len(r)-1)] - r[max(i-1, 0)] >= 0.2:
              cut = r[max(i-1, 0)] + dr
              break
          i = j

  return [ci] + [ti[k] for k in np.argsort(d) if d[k] <= cut]

def get_qm_mm_index(eles, xyz, cell):

  qm_indices = [180, 121, 76, 108, 84, 77, 6, 118, 142, 109]
  #qm_indices = first_shell(eles, xyz, cell, center="auto", target="F", cut=None, dr=0.05) 
  mm_indices = [i for i in range(len(eles)) if i not in qm_indices]

  return qm_indices, mm_indices

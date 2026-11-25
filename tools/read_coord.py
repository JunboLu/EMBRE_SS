#! /usr/env/bin python

import numpy as np
from pyscf.data import nist
from scipy.optimize import linear_sum_assignment

def read_last_frame(xyz_file):

  lines = open(xyz_file).readlines()
  nat = int(lines[0]); fs = nat + 2
  nfr = len(lines) // fs
  off = (nfr - 1) * fs + 2
  body = lines[off: off + nat]
  for l in body:
    assert len(l.split()) == 4, f"coord file is wrong"
  elements = [l.split()[0] for l in body]
  xyz = np.array([[float(x) for x in l.split()[1:4]] for l in body])

  return elements, xyz

def read_text(inp_file):

  with open(inp_file, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

  charge = int(lines[0])
  coord = '\n'.join(lines[1:])

  return charge, '\n' + coord + '\n'

def split_re_ligand(atoms):

    rare_earth = {'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                  'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Sc', 'Y'}

    re_atoms = []
    ligand_atoms = []

    for atom in atoms:
        element = atom[0]

        if element in rare_earth:
            re_atoms.append(atom)
        else:
            ligand_atoms.append(atom)

    return re_atoms, ligand_atoms, re_atoms[0][0]

def get_re_element(re_atoms):

  return re_atoms[0]

def get_atom_indices(atoms):

  rare_earth = {'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', \
                'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Sc', 'Y'}

  re_indices = []
  ligand_indices = []

  for i, atom_i in enumerate(atoms):
    element = atom_i[0]

    if element in rare_earth:
      re_indices.append(i)
    else:
      ligand_indices.append(i)

  return re_indices, ligand_indices

def get_box(cell_file):

    cell = []

    with open(cell_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if parts[0].lower() in ('a', 'b', 'c'):
                cell.append([float(x) for x in parts[1:4]])

    if len(cell) != 3:
        raise ValueError("cell_file must contain three lattice vectors (a, b, c).")

    return np.array(cell)

def get_bader_charge(acf_file='ACF.dat', xyz_file='Pr.xyz'):

    VALENCE = {'F': 7, 'La': 11, 'Ce': 12, 'Pr': 13, 'Nd': 14, 'Pm': 15, 'Sm': 16,
           'Eu': 17, 'Gd': 18, 'Tb': 29, 'Dy': 30, 'Ho': 31, 'Er': 32, 'Tm': 33,
           'Yb': 34, 'Lu': 35}

    # Read the last frame of XYZ (Angstrom)
    elements, xyz = read_last_frame(xyz_file)
    n_atoms = len(xyz)

    # Convert XYZ coordinates: Angstrom -> Bohr
    xyz = xyz / nist.BOHR  # nist.BOHR = 0.52917721092

    # Read ACF.dat (Bohr)
    acf_xyz = []
    bader_electrons = []

    with open(acf_file) as f:
        for line in f:
            fields = line.split()
            try:
                int(fields[0])
                acf_xyz.append([float(x) for x in fields[1:4]])
                bader_electrons.append(float(fields[4]))
            except (ValueError, IndexError):
                continue

    acf_xyz = np.array(acf_xyz)
    bader_electrons = np.array(bader_electrons)

    if len(acf_xyz) != n_atoms:
        raise ValueError(
            f"Atom numbers do not match: XYZ={n_atoms}, "
            f"ACF={len(acf_xyz)}"
        )

    # Match coordinates
    distance = np.linalg.norm(
        acf_xyz[:, None, :] - xyz[None, :, :], axis=2
    )

    row, col = linear_sum_assignment(distance)

    max_error = np.max(distance[row, col])

    if max_error > 0.01:
        raise ValueError(
            f"Coordinate matching failed! "
            f"Maximum error = {max_error:.6f} Bohr"
        )

    # Calculate charges in XYZ atom order
    charge = np.zeros(n_atoms)

    for i, j in zip(row, col):
        element = elements[j]

        if element not in VALENCE:
            raise ValueError(f"Unknown element: {element}")

        charge[j] = VALENCE[element] - bader_electrons[i]

    #print("Bader coordinate matching: OK")
    #print(f"Number of atoms: {n_atoms}")
    #print(f"Maximum matching error: {max_error:.8e} Bohr")

    return np.round(charge, 6)


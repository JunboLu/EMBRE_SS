#! /usr/env/bin python

def get_spin(re_element):

  spin_dict = {'Ce': 1, 'Pr': 2, 'Nd': 3, 'Pm': 4, 'Sm': 5, 'Eu': 6, 'Gd': 7, \
               'Tb': 6, 'Dy': 5, 'Ho': 4, 'Er': 3, 'Tm': 2, 'Yb': 1}

  return spin_dict[re_element]

def get_tot_elec_num(re_element):

  tot_elec_num_dict = {'Ce': 55, 'Pr': 56, 'Nd': 57, 'Pm': 58, 'Sm': 59, 'Eu': 60, 'Gd': 61, \
                       'Tb': 62, 'Dy': 63, 'Ho': 64, 'Er': 65, 'Tm': 66, 'Yb': 67}

  return tot_elec_num_dict[re_element]


def get_f_elec_num(re_element):

  f_elec_num_dict = {'Ce': 1, 'Pr': 2, 'Nd': 3, 'Pm': 4, 'Sm': 5, 'Eu': 6, 'Gd': 7, \
                     'Tb': 8, 'Dy': 9, 'Ho': 10, 'Er': 11, 'Tm': 12, 'Yb': 13}

  return f_elec_num_dict[re_element]

def get_cov_radius(re_element):

  ELE2RADIUS = {
    'La': 1.356, 'Ce': 1.336, 'Pr': 1.319, 'Nd': 1.303, 'Pm': 1.284,
    'Sm': 1.272, 'Eu': 1.260, 'Gd': 1.247, 'Tb': 1.235, 'Dy': 1.223,
    'Ho': 1.212, 'Er': 1.202, 'Tm': 1.192, 'Yb': 1.182, 'Lu': 1.172,
    'Y': 1.215,
    'F': 1.16, 'Cl': 1.67, 'Br': 1.82,
    'O': 1.22,
    }

  return ELE2RADIUS[re_element]

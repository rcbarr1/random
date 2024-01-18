#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:19:04 2023

@author: Reese
"""

import PyCO2SYS as pyco2

ta = [2325, 2425]
dic = [2175, 2350]

results = pyco2.sys(par1=ta, par2=dic, par1_type=1, par2_type=2,pressure=3000,
                    salinity=34.9,temperature=4)

del_pCO2 = results['pCO2'][1] - results['pCO2'][0]
del_HCO3 = results['HCO3'][1] - results['HCO3'][0]
del_CO3 = results['CO3'][1] - results['CO3'][0]
del_pH = results['pH'][1] - results['pH'][0]

print('')
print('∆pCO₂: {:.5} µatm'.format(del_pCO2))
print('∆HCO₃-: {:.5} µmol/kg'.format(del_HCO3))
print('∆CO₃²-: {:.5} µmol/kg'.format(del_CO3))
print('∆pH: {:.5} µmol/kg'.format(del_pH))




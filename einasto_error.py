from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy
import astropy
import astropy.constants as astroc
from scipy.optimize import fsolve
from numpy import *
import cosmolopy

from functions_fisher import *

"""Outputs uncertainty in three Einasto parameters (rho_2, r_2, alpha)
for a given cluster mass (M200), redshift (z_c) and M200 uncertainty """



"""fiducial cosmology"""
little_h_fid = 0.7
omega_M_fid = 0.3001
w_fid = -1.

"""define mass perc error"""
#statistical mass error on M200 increases if  cosmology is not fixed, as such, the following uncertainties in M200
# (for M200 = 4e14) are doubled (5% becomes 10% and so on)

mass_perc_error = 0.4 #specify stat error; percent error/100

"""define cluster"""
M200 = 4e14 #some fiducial mass
z_c = 0. #some fiducial redshift
conc =  concentration_meta(M200,z_c,little_h_fid)
R200 = (3*M200/((4*np.pi)*200.* rho_crit_z(z_c,w_fid,omega_M_fid,little_h_fid)))**(1./3.)

r_array_fit = np.arange(0.01,5,0.1)
ein_params_guess= [0.2,0.5,1e14]

"""uncertainty in NFW parameters"""
R200_uncertainty = (R200 / 3.) * (mass_perc_error)

M200_deltaM200_plus =  M200 + (M200 * mass_perc_error) 	
# print log10(M200_deltaM200_plus)
c200_deltac200_plus = concentration_meta(M200_deltaM200_plus,z_c,little_h_fid)
R200_deltaR200_plus = R200 + R200_uncertainty

""" einasto fit"""
end_r200, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200),2)+0.05) & (r_array_fit > round(find_nearest(r_array_fit,R200),2)-0.05) )[0]
rho_ein_params_array = scipy.optimize.curve_fit(rho_einasto,r_array_fit[0:end_r200+1], rho_nfw(r_array_fit[0:end_r200+1],M200,R200,conc),p0=ein_params_guess)

alpha =rho_ein_params_array[0][0]
r_2= rho_ein_params_array[0][1]
rho_2 = rho_ein_params_array[0][2]

"""einasto fit plus"""
end_r200_plus, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200_deltaR200_plus),2)+0.05) & (r_array_fit > round(find_nearest(r_array_fit,R200_deltaR200_plus),2)-0.05) )[0]
delta_plus_rho_ein_params_array = scipy.optimize.curve_fit(rho_einasto,r_array_fit[0:end_r200_plus+1], rho_nfw(r_array_fit[0:end_r200_plus+1],M200_deltaM200_plus,R200_deltaR200_plus,c200_deltac200_plus),p0=ein_params_guess)

delta_plus_alpha =delta_plus_rho_ein_params_array[0][0]
delta_plus_r_2= delta_plus_rho_ein_params_array[0][1]
delta_plus_rho_2 = delta_plus_rho_ein_params_array[0][2]

"""print sigma"""
sigma_alpha = np.abs(alpha-delta_plus_alpha)
sigma_r_2 = np.abs(r_2 - delta_plus_r_2)
sigma_rho_2 = np.abs(rho_2-delta_plus_rho_2)

print 'sigma(r_2)=',sigma_r_2, '[Mpc]'
print 'sigma(rho_2)=',sigma_rho_2/1e12,' 1e12 [Msun/Mpc3]'
print 'sigma(alpha)=',sigma_alpha

"""plot"""
# plt.plot(r_array_fit,rho_einasto(r_array_fit, alpha, r_2, rho_2),color='red',ls='--')
# plt.plot(r_array_fit,rho_einasto(r_array_fit, delta_plus_alpha, delta_plus_r_2, delta_plus_rho_2),color='red',ls=':')

# plt.plot(r_array_fit, rho_nfw(r_array_fit, M200, R200, conc),color='black',ls='--')
# plt.plot(r_array_fit, rho_nfw(r_array_fit, M200_deltaM200_plus, R200_deltaR200_plus, c200_deltac200_plus),color='black',ls=':')


# plt.loglog()

# plt.show()

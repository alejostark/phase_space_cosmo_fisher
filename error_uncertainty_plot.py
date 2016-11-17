from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pylab as p
import scipy
import astropy
import astropy.constants as astroc
import cosmolopy
import string
import sys
import numpy.random as npr
import scipy.optimize as optimize
import scipy.special as ss

from math import *
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from scipy.interpolate import interp1d
from scipy import ndimage
from astropy.table import Table, vstack
from astropy.io import fits
from causticpy import *
from scipy.optimize import fsolve
from numpy import *

from functions import *
from functions_fisher import *



def rho_einasto2(r,alpha,M,R):    
    n = 1/alpha
    r_2 = R * (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))

    rho_0 = M / 4. * np.pi   * (R**3.) * n * gamma_3n #kg
    rho_2 = rho_0 / np.exp(2.*n)

    return rho_2 * np.exp( (-2./alpha) * ( (r/r_2)**(alpha) -1 ) )




directory = '/Users/alejo/Dropbox/UMich/astro_research/paper2_data'
cluster_catalog = Table.read(directory+'/cluster_spectra/cluster_catalog.csv')


M200_array = (cluster_catalog['M200 1d14 [Msun]'] * 1e14)
M200_uncertainty_array= cluster_catalog['dM200 1d14 [Msun]'] *1e14

conc_array = cluster_catalog['c200']
R200_array = cluster_catalog['r200 [Mpc]']

z_c_array = cluster_catalog['z_c']


alpha_array = np.zeros_like(z_c_array)*0.
r_2_array = np.zeros_like(z_c_array)*0.
rho_2_array = np.zeros_like(z_c_array)*0.

delta_alpha_minus_array = np.zeros_like(z_c_array)*0.
delta_r_2_minus_array = np.zeros_like(z_c_array)*0.
delta_rho_2_minus_array = np.zeros_like(z_c_array)*0.

delta_alpha_plus_array = np.zeros_like(z_c_array)*0.
delta_r_2_plus_array = np.zeros_like(z_c_array)*0.
delta_rho_2_plus_array = np.zeros_like(z_c_array)*0.


little_h = 0.7
omega_M = 0.3

for i in range(0,len(M200_array)):
	M200 = M200_array[i]
	R200 = R200_array[i]
	conc = conc_array[i]
	z_c  = z_c_array[i]

	r_array_fit = np.arange(0,5,0.1)

	end_r200, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200),2)+0.05) & (r_array_fit > round(find_nearest(r_array_fit,R200),2)-0.05) )
	ein_params_guess= [0.2,1e12,0.2]
	rho_ein_params_array = optimize.curve_fit(rho_einasto2,r_array_fit[1:end_r200], rho_nfw(r_array_fit[1:end_r200],M200,R200,conc),p0=ein_params_guess)

	alpha =rho_ein_params_array[0][0]
	r_2= rho_ein_params_array[0][1]
	rho_2 = rho_ein_params_array[0][2]

	delta_M_WL =  M200 + M200_uncertainty_array[i] 
	# delta_M_WL =  M200 + (M200 *.10) 

	delta_conc = concentration_meta(delta_M_WL,z_c,little_h)
	delta_R= (3*delta_M_WL/ (4.*np.pi*200.*rho_crit_z(z_c,omega_M,little_h)) )**(1./3.)

	r_array_fit = np.arange(0.01,5,0.1)
	delta_ein_params_guess= [0.2,0.5,1e14]
	end_r200, = np.where( (r_array_fit < round(find_nearest(r_array_fit,delta_R),2)+0.05) & (r_array_fit > round(find_nearest(r_array_fit,delta_R),2)-0.05) )

	###Generate new v_esc_nfw_einasto_q profile with %error on M200 ###
	delta_rho_ein_params_array = optimize.curve_fit(rho_einasto2,r_array_fit[0:end_r200+1], rho_nfw(r_array_fit[0:end_r200+1],delta_M_WL,delta_R,delta_conc),p0=delta_ein_params_guess)

	delta_alpha =delta_rho_ein_params_array[0][0]
	delta_r_2= delta_rho_ein_params_array[0][1]
	delta_rho_2 = delta_rho_ein_params_array[0][2]

	alpha_array[i]= alpha
	r_2_array[i]= r_2
	rho_2_array[i]= rho_2

	delta_alpha_plus_array[i]= delta_alpha
	delta_r_2_plus_array[i]= delta_r_2
	delta_rho_2_plus_array[i]= delta_rho_2

	delta_alpha_minus_array[i]= delta_alpha_minus
	delta_r_2_minus_array[i]= delta_r_2_minus
	delta_rho_2_minus_array[i]= delta_rho_2_minus





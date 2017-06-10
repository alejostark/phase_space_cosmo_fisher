from __future__ import division
from functions_fisher import *
from math import *
from scipy.misc import derivative as derivative
from sympy import *
from sympy.matrices import *
from numpy import meshgrid as coord
from astropy import units as u
from matplotlib import cm
import scipy.special as ss
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as astroc
import matplotlib
from astropy.table import Table, vstack


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

GOAL: Calculate and plot Fisher matrix 2-dimensional marginalized likelihood for the escape velocity profile:

							v_esc(r,z_c,cluster_params,cosmo_params) 

Where,

'r' is a radius array defined below and 'z_c' is the cluster redshift

'cluster_params' refers to beta, alpha, rho-2,r-2 (last three from Einasto profile and beta = anisotropy parameter)

'cosmo_params' are divided into two different cases, you need to pick pick a 'case' to make inverted F matrix (ie the G matrix)

(1) flat universe(omegaM=1-omegaDE, w, H0)
    1a) constant w (omegaM, w, H0)... case = 'flat'
    1b) evolving w(z) w/ h prior (omegaM, w0, wa, H0)) ... case= 'w_z'
    1c) evolving w(z) (omegaM, w0, wa, H0) ... case = 'w_z_riess16_h'
    1d) evolving w(z) w/ fixed h (omegaM, w0, wa) ... case = 'w_z_fixed_h'
    
(2) non-flat universe w/ w=-1 
    2a) no Riess prior (omegaM, omega_L, H0) ... case = 'non_flat'
	2b) w/ Riess prior (omegaM, omega_L, H0) ... case = 'non_flat_riess_prior'

In terms of cosmo parames,  in all 3 cases we marginalize over H0 to attain a 2-dim likelihood. 
For #2 we marginalize over H0 and omegaM.

INSTRUCTIONS:
See commented out code in the last section of fisher_matrix.py (under "user input")
(1) Pick one of the cases to run below
(2) Specify the number of clusters and redshift range for those clusters
(3) Specify uncertainties on the 4 cluster parameters and observed escape velocity "edge." You may also want to change the cluster uncertainty parameters 
(see function below) if you want to change the values going into the prior matrix. You can re-calculate these for any mass percent error with einasto_error.py

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""
FIDUCIAL VALUES to evaluate Fisher matrix derivatives
"""""""""""""""""""""""""""""""""""""""""""""""""""""
radius_array = np.linspace(0.5,2.5,14).round(3) #specify radius array for profiles. used in v_esc(r) funcs below.

#alpha_fid, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,w0_fid,wa_fid

""" equivalent to M200=4e14 at z=0; use einasto_error.py to map"""
alpha_fid = 0.1984
rho_2_fid = 105129885699836.28/1e14
r_2_fid =  0.497
beta_fid = 0.145 

Omega_M_fid = 0.3001
little_h_fid = 0.7
w_fid = -1.

Omega_DE_fid = 0.7

w0_fid = -1.
wa_fid = 0.

lil_Omega_M_fid = Omega_M_fid * little_h_fid**2.
lil_Omega_DE_fid  = Omega_DE_fid * little_h_fid**2.

z_t = z_trans([w_fid,Omega_M_fid],'flat')

"""""""""""""""""""""""
CLUSTER PARAMETER UNCERTAINTIES 
"""""""""""""""""""""""
def cluster_uncertainty_params(case):
	#units of r_2 in Mpc ; units of rho_2 in Msun/Mpc^3 ;alpha is unitless

	if case == '5pct_none':
		#UNCERTAINTIES used assuming we're stacking clusters
		#and we're **NOT** FIXING cosmology, 5% -> 10%

		######Prior from Lokas paper on SINGLE beta######
		sigma_beta_squared =  (0.02)**2. 

		######5% mass error######
		sigma_r2_squared = (0.031413740507)**2.
		sigma_rho2_squared = (5.8872905321e12/1e14)**2.
		sigma_alpha_squared = (0.00248973353699)**2. 

		sigma_squared_list = [sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared]

		cluster_edge_unc = np.sqrt(50**2 + (1000 * 0.15/2)**2 )

	elif case == '20pct_none':
		#UNCERTAINTIES used assuming we're using 20% scatter on M200
		#and we're **NOT** FIXING cosmology, 20% -> 40%

		######Prior from Lokas paper on SINGLE beta######
		sigma_beta_squared =  (0.5)**2.  

		####40% mass error#####
		sigma_r2_squared = ( 0.134207730737)**2.
		sigma_rho2_squared = (23.5893129936e12/1e14)**2.
		sigma_alpha_squared = (0.00961946729343)**2.

		sigma_squared_list = [sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared]
		
		cluster_edge_unc = np.sqrt(50**2 + 50**2 + (1000*0.25/2)**2.) 

	elif case == '40pct_none':
		#UNCERTAINTIES used assuming we're using 40% scatter on M200
		#and we're **NOT** FIXING cosmology, 40% -> 80%

		####Prior from Lokas paper on SINGLE beta######
		sigma_beta_squared =  (0.5)**2.  

		####40% mass error#####
		sigma_r2_squared = (  0.291313467809)**2. 
		sigma_rho2_squared = (43.9040001563e12/1e14)**2. 
		sigma_alpha_squared = (0.0181365400778)**2.

		sigma_squared_list = [sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared]
		
		cluster_edge_unc = np.sqrt(50**2 + 50**2 + (1000*0.25/2)**2.)

	elif case == '40pct_riess':
		#UNCERTAINTIES used assuming we're using stat. 40% scatter on M200
		#and we're applyign Riess et al '16 prior on H0
		#that means 40pct_none (80% error) goes down to 40%

		######Prior from Lokas paper on SINGLE beta######
		sigma_beta_squared =  (0.5)**2.  

		####40% mass error#####
		sigma_r2_squared = ( 0.134207730737)**2.
		sigma_rho2_squared = (23.5893129936e12/1e14)**2.
		sigma_alpha_squared = (0.00961946729343)**2.

		sigma_squared_list = [sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared]
		
		cluster_edge_unc = np.sqrt(50**2 + 50**2 + (1000*0.25/2)**2.)

	return sigma_squared_list, cluster_edge_unc

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
FISHER MATRIX :Run loop of derivatives for Nclus and make Fisher matrix for three cases: 
 flat (omegaM, w, H0); flat (omegaM, w0, wa, H0) ;non_flat (omegaM,omegaL, H0)
  and plot.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def calculate_derivatives(z_c_array,case):

	dx_deriv = 1e-10	#derivative

	if case == 'flat':
		#initiate lists to attach derivatives for all the clusters
		derivatives_COSMOparam_list = []
		derivatives_CLUSTERparam_list = []

		#loop over all clusters
		print 'calculating derivatives...'
		for redshift_index in range(0,len(z_c_array)):

			theta_array = radius_array / D_A(z_c_array[redshift_index],[w_fid, Omega_M_fid, little_h_fid],'flat')

			### cosmology derivatives ###
			dv_dOmegaM = derivative(lambda x: v_esc_theory_flat(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, beta_fid, x,little_h_fid,w_fid), Omega_M_fid,dx_deriv)
			dv_dh      = derivative(lambda x: v_esc_theory_flat(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid, x ,w_fid), little_h_fid,dx_deriv)
			dv_dw      = derivative(lambda x: v_esc_theory_flat(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,x), w_fid,dx_deriv)

			derivatives_COSMOparam_list.append(dv_dOmegaM)
			derivatives_COSMOparam_list.append(dv_dh)
			derivatives_COSMOparam_list.append(dv_dw)

			### beta derivative ###
			dv_dbeta  = derivative(lambda x: v_esc_theory_flat(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, x, Omega_M_fid,little_h_fid,w_fid), beta_fid,dx_deriv)

			### weak lensing derivatives ###
			dv_dalpha = derivative(lambda x: v_esc_theory_flat(theta_array,z_c_array[redshift_index],x, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,w_fid), alpha_fid,dx_deriv)
			dv_dr_2   = derivative(lambda x: v_esc_theory_flat(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, x, beta_fid, Omega_M_fid,little_h_fid,w_fid), r_2_fid,dx_deriv)
			dv_drho_2 = derivative(lambda x: v_esc_theory_flat(theta_array,z_c_array[redshift_index],alpha_fid, x, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,w_fid), rho_2_fid,dx_deriv)

			derivatives_CLUSTERparam_list.append(dv_dbeta)
			derivatives_CLUSTERparam_list.append(dv_dalpha)
			derivatives_CLUSTERparam_list.append(dv_dr_2)
			derivatives_CLUSTERparam_list.append(dv_drho_2)

		#array of length n clusters by kth radial bins of v_esc deriv. w.r.t. COSMO params
		dv_dOmegaM_array = np.array(derivatives_COSMOparam_list[0::3]) 
		dv_dh_array = np.array(derivatives_COSMOparam_list[1::3])
		dv_dw_array = np.array(derivatives_COSMOparam_list[2::3])

		#array of length n clusters by kth radial bins of v_esc deriv. w.r.t. CLUSTER params
		dv_dbeta_array = np.array(derivatives_CLUSTERparam_list[0::4])
		dv_dalpha_array = np.array(derivatives_CLUSTERparam_list[1::4])
		dv_dr_2_array = np.array(derivatives_CLUSTERparam_list[2::4])
		dv_drho_2_array = np.array(derivatives_CLUSTERparam_list[3::4])

		#put together into a 7 x Nclus x Nradial_bins list
		return dv_dOmegaM_array, dv_dw_array,dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array

	elif case == 'non_flat':

		#initiate lists to attach derivatives for all the clusters
		derivatives_COSMOparam_list = []
		derivatives_CLUSTERparam_list = []

		#loop over all clusters
		print 'calculating derivatives...'

		for redshift_index in range(0,len(z_c_array)):

			theta_array = radius_array / D_A(z_c_array[redshift_index],[Omega_DE_fid, Omega_M_fid, little_h_fid],'nonflat')

			### cosmology derivatives ###
			dv_dOmegaM = derivative(lambda x: v_esc_theory_non_flat(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, beta_fid,           x,little_h_fid,   Omega_DE_fid), Omega_M_fid,dx_deriv)
			dv_domegaDE= derivative(lambda x: v_esc_theory_non_flat(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,              x), Omega_DE_fid, dx_deriv)
			dv_dh      = derivative(lambda x: v_esc_theory_non_flat(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid,          x ,   Omega_DE_fid), little_h_fid,dx_deriv)

			derivatives_COSMOparam_list.append(dv_dOmegaM)
			derivatives_COSMOparam_list.append(dv_domegaDE)
			derivatives_COSMOparam_list.append(dv_dh)

			### beta derivative ###
			dv_dbeta  = derivative(lambda x: v_esc_theory_non_flat(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, x, Omega_M_fid,little_h_fid,Omega_DE_fid), beta_fid,dx_deriv)

			### weak lensing derivatives ###
			dv_dalpha = derivative(lambda x: v_esc_theory_non_flat(theta_array,z_c_array[redshift_index],x, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,Omega_DE_fid), alpha_fid,dx_deriv)
			dv_dr_2   = derivative(lambda x: v_esc_theory_non_flat(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, x, beta_fid, Omega_M_fid,little_h_fid,Omega_DE_fid), r_2_fid,dx_deriv)
			dv_drho_2 = derivative(lambda x: v_esc_theory_non_flat(theta_array,z_c_array[redshift_index],alpha_fid, x, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,Omega_DE_fid), rho_2_fid,dx_deriv)

			derivatives_CLUSTERparam_list.append(dv_dbeta)
			derivatives_CLUSTERparam_list.append(dv_dalpha)
			derivatives_CLUSTERparam_list.append(dv_dr_2)
			derivatives_CLUSTERparam_list.append(dv_drho_2)

		#array of length n clusters by kth radial bins of v_esc deriv. w.r.t. COSMO params
		dv_dOmegaM_array = np.array(derivatives_COSMOparam_list[0::3]) 
		dv_domegaDE_array = np.array(derivatives_COSMOparam_list[1::3])
		dv_dh_array = np.array(derivatives_COSMOparam_list[2::3])

		#array of length n clusters by kth radial bins of v_esc deriv. w.r.t. CLUSTER params
		dv_dbeta_array = np.array(derivatives_CLUSTERparam_list[0::4])
		dv_dalpha_array = np.array(derivatives_CLUSTERparam_list[1::4])
		dv_dr_2_array = np.array(derivatives_CLUSTERparam_list[2::4])
		dv_drho_2_array = np.array(derivatives_CLUSTERparam_list[3::4])

		return dv_dOmegaM_array, dv_domegaDE_array,dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array

	elif case == 'w_z':
		derivatives_COSMOparam_list = []
		derivatives_CLUSTERparam_list = []

		print 'calculating derivatives...'
		for redshift_index in range(0,len(z_c_array)):

			theta_array = radius_array / D_A(z_c_array[redshift_index],[w0_fid, wa_fid, Omega_M_fid, little_h_fid],'w_z')

			### cosmology derivatives ###
			dv_dOmegaM = derivative(lambda x: v_esc_theory_w_z(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, beta_fid, x,little_h_fid,w0_fid,wa_fid), Omega_M_fid,dx_deriv)
			dv_dw0      = derivative(lambda x: v_esc_theory_w_z(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,x,wa_fid), w0_fid,dx_deriv)
			dv_dwa      = derivative(lambda x: v_esc_theory_w_z(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,w0_fid,x), wa_fid,dx_deriv)
			dv_dh      = derivative(lambda x: v_esc_theory_w_z(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid, x ,w0_fid,wa_fid), little_h_fid,dx_deriv)

			derivatives_COSMOparam_list.append(dv_dOmegaM)
			derivatives_COSMOparam_list.append(dv_dw0)
			derivatives_COSMOparam_list.append(dv_dwa)
			derivatives_COSMOparam_list.append(dv_dh)

			### beta derivative ###
			dv_dbeta  = derivative(lambda x: v_esc_theory_w_z(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, r_2_fid, x, Omega_M_fid,little_h_fid,w0_fid,wa_fid), beta_fid,dx_deriv)

			### weak lensing derivatives ###
			dv_dalpha = derivative(lambda x: v_esc_theory_w_z(theta_array,z_c_array[redshift_index],x, rho_2_fid, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,w0_fid,wa_fid), alpha_fid,dx_deriv)
			dv_dr_2   = derivative(lambda x: v_esc_theory_w_z(theta_array,z_c_array[redshift_index],alpha_fid, rho_2_fid, x, beta_fid, Omega_M_fid,little_h_fid,w0_fid,wa_fid), r_2_fid,dx_deriv)
			dv_drho_2 = derivative(lambda x: v_esc_theory_w_z(theta_array,z_c_array[redshift_index],alpha_fid, x, r_2_fid, beta_fid, Omega_M_fid,little_h_fid,w0_fid,wa_fid), rho_2_fid,dx_deriv)

			derivatives_CLUSTERparam_list.append(dv_dbeta)
			derivatives_CLUSTERparam_list.append(dv_dalpha)
			derivatives_CLUSTERparam_list.append(dv_dr_2)
			derivatives_CLUSTERparam_list.append(dv_drho_2)

		#array of length n clusters by kth radial bins of v_esc deriv. w.r.t. COSMO params
		dv_dOmegaM_array = np.array(derivatives_COSMOparam_list[0::4]) 
		dv_dw0_array = np.array(derivatives_COSMOparam_list[1::4])
		dv_dwa_array = np.array(derivatives_COSMOparam_list[2::4])
		dv_dh_array = np.array(derivatives_COSMOparam_list[3::4])

		#array of length n clusters by kth radial bins of v_esc deriv. w.r.t. CLUSTER params
		dv_dbeta_array = np.array(derivatives_CLUSTERparam_list[0::4])
		dv_dalpha_array = np.array(derivatives_CLUSTERparam_list[1::4])
		dv_dr_2_array = np.array(derivatives_CLUSTERparam_list[2::4])
		dv_drho_2_array = np.array(derivatives_CLUSTERparam_list[3::4])

		return dv_dOmegaM_array, dv_dw0_array,dv_dwa_array, dv_dh_array,dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array

def make_G_matrix(z_array, case,sigma_squared_list,cluster_edge_unc):

	#cosmology prior is nil here, just priors on cluster params
	sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared  = sigma_squared_list

	#delete clusters within 1% to the "left" of transition redshift
	sel, =np.where( (z_array > (z_t - z_t*.01)) & (z_array <= z_t))
	z_c_array_del = np.delete(z_array, sel)

	#add clusters immediately to left of trans redshift
	add_z_array = np.array([(z_t - z_t*.01)]*len(sel))

	if len(add_z_array) > 0:
		print '****replaced ', len(add_z_array), ' cluster(s)!***'
	else:
		pass

	z_c_array = np.insert(z_c_array_del,1,add_z_array)

	if case == 'flat':
		""""""""""""""""""""
		""""read in derivatives"""
		""""""""""""""""""""
		dv_dOmegaM_array, dv_dw_array,dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array  = calculate_derivatives(z_c_array,'flat') 
	
		#put together into a 7 x Nclus x Nradial_bins list
		derivs_list = [dv_dOmegaM_array, dv_dw_array,dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array]

		""""""""""""""""""""
		"""Fisher matrix """
		""""""""""""""""""
		print 'making matrix..'

		N_cosmo = 3
		F = make_fisher_matrix(derivs_list,cluster_edge_unc,N_cosmo)

		""""""""""""""""""
		"""Fisher prior"""
		""""""""""""""""""
		#Fprior
		N_clus = len(dv_dbeta_array)
		N_dim = 4* N_clus + N_cosmo

		F_prior = make_prior_matrix(sigma_squared_list,N_cosmo,N_clus)

		""""""""""""""""""
		"""Fisher total"""
		""""""""""""""""""
		F_tot = F + F_prior

		F_tot  = (np.array(np.array(F_tot),dtype='float'))
		print 'inverting F_tot matrix'
		F_inv_tot = np.linalg.inv(F_tot)

		G_inv_tot = F_inv_tot[0:2, 0:2]
		G_matrix_tot = np.linalg.inv(G_inv_tot)

		print 'marginalized sigma(omega_M) >= ', np.sqrt(G_inv_tot[0][0])
		print 'marginalized  sigma(w) >= ', np.sqrt(G_inv_tot[1][1])
			
	elif case == 'w_z':
		
		""""""""""""""""""""
		""""read in derivatives"""
		""""""""""""""""""""
		dv_dOmegaM_array, dv_dw0_array,dv_dwa_array, dv_dh_array,dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array = calculate_derivatives(z_c_array,'w_z') 

		#put together into a 8 x Nclus x Nradial_bins list
		derivs_list = [dv_dOmegaM_array, dv_dw0_array,dv_dwa_array, dv_dh_array,dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array]

		""""""""""""""""""""
		"""Fisher matrix """
		""""""""""""""""""
		N_cosmo = 4 #omegaM, w0, wa, H0
		N_clus = len(dv_dbeta_array)
		N_dim = 4* N_clus + N_cosmo

		print 'making matrix..'
		F = make_fisher_matrix(derivs_list,cluster_edge_unc,N_cosmo)
		
		""""""""""""""""""
		"""Fisher prior"""
		""""""""""""""""""

		F_prior = make_prior_matrix(sigma_squared_list,N_cosmo,N_clus)

		""""""""""""""""""
		"""Fisher total"""
		""""""""""""""""""
		F_tot = F + F_prior

		F_tot  = (np.array(np.array(F_tot),dtype='float'))
		print 'inverting F_tot matrix'
		F_inv_tot = np.linalg.inv(F_tot)

		#w0-wa submatrix marginalized
		G_inv_tot = F_inv_tot[1:3, 1:3]
		G_matrix_tot = np.linalg.inv(G_inv_tot)

		#oM-wa submatrix marginalized
		G_inv_tot_oM_wa = F_inv_tot[0:2, 0:2]

		print 'marginalized sigma(w0) >= ', np.sqrt(G_inv_tot[0][0])
		print 'marginalized  sigma(wa) >= ', np.sqrt(G_inv_tot[1][1])

		print 'marginalized  sigma(oM) >= ', np.sqrt(G_inv_tot_oM_wa[0][0])


		########################################
		# calculate pivot redshift
		#######################################

		sigma_wa_squared = (G_inv_tot[1][1])**2.
		sigma_w0_wa = G_inv_tot[0][1]

		z_p = -sigma_w0_wa / (sigma_w0_wa + sigma_wa_squared)
		print 'Pivot redshift =', z_p

	elif case == 'w_z_riess16_h':
		
		""""""""""""""""""""
		""""read in derivatives"""
		""""""""""""""""""""
		dv_dOmegaM_array, dv_dw0_array,dv_dwa_array, dv_dh_array,dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array = calculate_derivatives(z_c_array,'w_z') 

		#put together into a 8 x Nclus x Nradial_bins list
		derivs_list = [dv_dOmegaM_array, dv_dw0_array,dv_dwa_array, dv_dh_array,dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array]

		""""""""""""""""""""
		"""Fisher matrix """
		""""""""""""""""""
		N_cosmo = 4 #omegaM, w0, wa, H0
		N_clus = len(dv_dbeta_array)
		N_dim = 4* N_clus + N_cosmo

		print 'making matrix..'
		F = make_fisher_matrix(derivs_list,cluster_edge_unc,N_cosmo)
		
		""""""""""""""""""
		"""Fisher prior"""
		""""""""""""""""""

		F_prior = make_prior_matrix_cosmo_prior(sigma_squared_list,N_cosmo,N_clus)

		""""""""""""""""""
		"""Fisher total"""
		""""""""""""""""""
		F_tot = F + F_prior

		F_tot  = (np.array(np.array(F_tot),dtype='float'))
		print 'inverting F_tot matrix'
		F_inv_tot = np.linalg.inv(F_tot)

		G_inv_tot = F_inv_tot[1:3, 1:3]
		G_matrix_tot = np.linalg.inv(G_inv_tot)

	
		print 'marginalized sigma(w0) >= ', np.sqrt(G_inv_tot[0][0])
		print 'marginalized  sigma(wa) >= ', np.sqrt(G_inv_tot[1][1])

	elif case == 'non_flat':

		""""""""""""""""""""
		""""read in derivatives"""
		""""""""""""""""""""
		dv_dOmegaM_array, dv_domegaDE_array, dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array = calculate_derivatives(z_c_array,'non_flat') 

		#put together into a 7 x Nclus x Nradial_bins list
		derivs_list = [dv_dOmegaM_array, dv_domegaDE_array,dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array]

		""""""""""""""""""""
		"""Fisher matrix """
		""""""""""""""""""
		print 'making matrix..'

		N_cosmo = 3 #Number of cosmological parameters we're probing, parameters related to cluster are 4: 3 from Einasto the other is beta
		F = make_fisher_matrix(derivs_list,cluster_edge_unc,N_cosmo)
		
		""""""""""""""""""
		"""Fisher prior"""
		""""""""""""""""""
		#Fprior
		N_clus = len(dv_dbeta_array)
		N_dim = 4* N_clus + N_cosmo

		F_prior = make_prior_matrix(sigma_squared_list,N_cosmo,N_clus)

		""""""""""""""""""
		"""Fisher total"""
		""""""""""""""""""
		F_tot = F + F_prior

		F_tot  = (np.array(np.array(F_tot),dtype='float'))
		print 'inverting F_tot matrix'
		F_inv_tot = np.linalg.inv(F_tot)

		G_inv_tot = F_inv_tot[0:2, 0:2]
		G_matrix_tot = np.linalg.inv(G_inv_tot)

		###PLOTS####
		print 'marginalized sigma(omega_M) >= ', np.sqrt(G_inv_tot[0][0])
		print 'marginalized  sigma(omega_Lambda) >= ', np.sqrt(G_inv_tot[1][1])

	elif case == 'non_flat_riess_prior':
		""""""""""""""""""""
		""""read in derivatives"""
		""""""""""""""""""""
		dv_dOmegaM_array, dv_domegaDE_array, dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array = calculate_derivatives(z_c_array,'non_flat') 

		#put together into a 7 x Nclus x Nradial_bins list
		derivs_list = [dv_dOmegaM_array, dv_domegaDE_array,dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array]

		""""""""""""""""""""
		"""Fisher matrix """
		""""""""""""""""""
		print 'making matrix..'

		N_cosmo = 3 #Number of cosmological parameters we're probing, parameters related to cluster are 4: 3 from Einasto the other is beta
		F = make_fisher_matrix(derivs_list,cluster_edge_unc,N_cosmo)
		
		""""""""""""""""""
		"""Fisher prior"""
		""""""""""""""""""
		#Fprior
		N_clus = len(dv_dbeta_array)
		N_dim = 4* N_clus + N_cosmo

		# F_prior = make_prior_matrix(sigma_squared_list,N_cosmo,N_clus)
		F_prior = make_prior_matrix_cosmo_prior(sigma_squared_list,N_cosmo,N_clus)

		""""""""""""""""""
		"""Fisher total"""
		""""""""""""""""""
		F_tot = F + F_prior

		F_tot  = (np.array(np.array(F_tot),dtype='float'))
		print 'inverting F_tot matrix'
		F_inv_tot = np.linalg.inv(F_tot)

		G_inv_tot = F_inv_tot[0:2, 0:2]
		G_matrix_tot = np.linalg.inv(G_inv_tot)

		

		###PLOTS####
		print 'marginalized sigma(omega_M) >= ', np.sqrt(G_inv_tot[0][0])
		print 'marginalized  sigma(omega_Lambda) >= ', np.sqrt(G_inv_tot[1][1])

	#calculate condition number. should be < 12 for stable inverversion.
 	print 'log10(cond. number of Ftot) = ', np.log10(np.linalg.cond(F_tot))
	#return Matrix
	return Matrix(G_matrix_tot)


"""""""""""
PLOT 2D CONTOURS 
"""""""""

def plot_2d_contours_from_G(G_matrix_tot,case):

	#generate and plot contours
	f_1sig = 0.434 #68% CL ellipse
	f_2sig = 0.167 #95% CL ellipse
	f_3sig = 0.072 #95% CL ellipse

	if case == 'flat':
		###PLOTS####
		omega_M_array = np.arange(-1,1.,3e-3)
		w_array = np.arange(-5,2, 8e-3)

		x, y = coord(omega_M_array, w_array)# return coordinate matrices from coordinate vectors

		z_tot = G_matrix_tot[0]*x**2 + 2.0*G_matrix_tot[1]*(x*y) + G_matrix_tot[3]*(y**2)

		plt.contourf(x+.3, y-1.0, z_tot,  [1/f_2sig,1/f_1sig], colors='firebrick') #marginalized
		plt.contourf(x+.3, y-1.0, z_tot,  [1/f_1sig,1/f_2sig], colors='red') #marginalized

		plt.xlabel('$\Omega_M$',fontsize=20)
		plt.ylabel('$w$',fontsize=20)

		plt.xlim(0,1)
		plt.show()

	elif case == 'w_z':
		#marginalize over omegaM,h and other parameters to get w0-wa plane
		###PLOTS####
		w0_array = np.arange(-5,2, 1e-2)
		wa_array = np.arange(-5,5, 1e-2)

		x, y = coord(w0_array,wa_array)# return coordinate matrices from coordinate vectors

		z_tot = G_matrix_tot[0]*x**2 + 2.0*G_matrix_tot[1]*(x*y) + G_matrix_tot[3]*(y**2)

		plt.contour(x-1., y, z_tot,  [1/f_1sig,1/f_2sig], colors='red',ls='--',linewidths=1) #marginalized

		plt.xlabel('$w_0$',fontsize=20)
		plt.ylabel('$w_a$',fontsize=20)
		plt.show()

	elif case == 'w_z_riess16_h':
		#marginalize over omegaM,h and other parameters to get w0-wa plane
		###PLOTS####
		w0_array = np.arange(-5,2, 1e-2)
		wa_array = np.arange(-5,5, 1e-2)

		x, y = coord(w0_array,wa_array)# return coordinate matrices from coordinate vectors

		z_tot = G_matrix_tot[0]*x**2 + 2.0*G_matrix_tot[1]*(x*y) + G_matrix_tot[3]*(y**2)

		plt.contour(x-1., y, z_tot,  [1/f_1sig,1/f_2sig], colors='red',ls='--',linewidths=1) #marginalized

		plt.xlabel('$w_0$',fontsize=20)
		plt.ylabel('$w_a$',fontsize=20)
		plt.show()

	elif case == 'w_z_fixed_h':
		#marginalize over omegaM,h and other parameters to get w0-wa plane
		###PLOTS####
		w0_array = np.arange(-5,2, 1e-2)
		wa_array = np.arange(-5,5, 1e-2)

		x, y = coord(w0_array,wa_array)# return coordinate matrices from coordinate vectors

		z_tot = G_matrix_tot[0]*x**2 + 2.0*G_matrix_tot[1]*(x*y) + G_matrix_tot[3]*(y**2)

		plt.contour(x-1., y, z_tot,  [1/f_1sig,1/f_2sig], colors='red',ls='--',linewidths=1) #marginalized

		plt.xlabel('$w_0$',fontsize=20)
		plt.ylabel('$w_a$',fontsize=20)
		plt.show()

	elif case == 'non_flat':
		omega_M_array = np.arange(-1,1.,2e-3)
		omega_DE_array = np.arange(-1,1.3,2e-3)
		h_array = np.arange(.5, .8, 1e-2) 

		x, y = coord(omega_M_array, omega_DE_array)# return coordinate matrices from coordinate vectors
		z = G_matrix_tot[0]*x**2 + 2.0*G_matrix_tot[1]*(x*y) + G_matrix_tot[3]*(y**2)

		plt.contourf(x+0.3, y+0.7, z,  [1/f_2sig, 1/f_1sig] , colors='black') #marginalized
		plt.contourf(x+0.3, y+0.7, z, [1/f_1sig,1/f_2sig] , colors='gray') #marginalized

		plt.xlabel('$\Omega_M $',fontsize=20)
		plt.ylabel('$\Omega_{\Lambda}$',fontsize=20)

		plt.show()

	elif case == 'non_flat_riess_prior':
		omega_M_array = np.arange(-1,1.,2e-3)
		omega_DE_array = np.arange(-1,1.3,2e-3)

		x, y = coord(omega_M_array, omega_DE_array)# return coordinate matrices from coordinate vectors
		z = G_matrix_tot[0]*x**2 + 2.0*G_matrix_tot[1]*(x*y) + G_matrix_tot[3]*(y**2)

		plt.contour(x+0.3, y+0.7, z,  [1/f_2sig,1/f_2sig], colors='red',linewidths=1) #marginalized

		plt.xlabel('$\Omega_M $',fontsize=20)
		plt.ylabel('$\Omega_{\Lambda}$',fontsize=20)

		plt.show()

"""""""""""
OTHER PLOTS 
"""""""""

def plot_derivatives_cluster(z_c_array):

	dv_dOmegaM_array, dv_domegaDE_array,dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array = calculate_derivatives(z_c_array,'non_flat') 

	cmap = cm.Spectral_r
	colors = [cmap(i) for i in np.linspace(0, 1, len(z_c_array)-1)]

	fig, ax = plt.subplots(2,2)

	plot_lines_beta = [ ax[0,0].plot(radius_array,dv_dbeta_array[i], color=color) for i, color in enumerate(colors, start=1)]
	ax[0,0].set_ylabel(r'$\partial v_{esc}(r,z)/\partial \beta$ [km/s]',fontsize=15)
	ax[0,0].set_xlabel('radius [Mpc]',fontsize=15)

	plot_lines_alpha = [ ax[0,1].plot(radius_array,dv_dalpha_array[i], color=color) for i, color in enumerate(colors, start=1)]
	ax[0,1].set_ylabel(r'$\partial v_{esc}(r,z)/\partial \alpha$ [km/s]',fontsize=15)
	ax[0,1].set_xlabel('radius [Mpc]',fontsize=15)

	plot_lines_h = [ ax[1,0].plot(radius_array,dv_dr_2_array[i], color=color) for i, color in enumerate(colors, start=1)]
	ax[1,0].set_ylabel(r'$\partial v_{esc}(r,z)/\partial r_{-2}$ [km/s/Mpc]',fontsize=15)
	ax[1,0].set_xlabel('radius [Mpc]',fontsize=15)

	plot_lines_omegaL = [ ax[1,1].plot(radius_array,dv_drho_2_array[i], color=color) for i, color in enumerate(colors, start=1)]
	# ax[1,1].set_ylabel('$\partial v_{esc}(r,z_n)/\partial h$ [km/s]',fontsize=18)
	ax[1,1].set_ylabel(r'$\partial v_{esc}(r,z)/\partial \rho_{-2}$'+'[km/s]/['+r'$M_{\odot}/$'+'Mpc'+r'$^3$'+']',fontsize=15)
	ax[1,1].set_xlabel('radius [Mpc]',fontsize=15)

	#colorbar
	norm = matplotlib.colors.Normalize(vmin=np.min(z_c_array),vmax=np.max(z_c_array))
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(z_c_array), vmax=np.max(z_c_array)))
	sm._A = [] 	# fake up the array of the scalar mappable. Urgh...


	ax[0,0].set_xlim(0.5,2.5)
	# ax[0,0].set_ylim(-100,3000)

	ax[0,1].set_xlim(0.5,2.5)
	# ax[0,1].set_ylim(-50,200)

	ax[1,0].set_xlim(0.5,2.5)
	# ax[1,0].set_ylim(-300,50)

	ax[1,1].set_xlim(0.5,2.5)
	# ax[1,1].set_ylim(-1500,100)

	cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])

	plt.colorbar(sm,label='redshift',cax=cbar_ax)

	# plt.xlim(np.min(radius_array),np.max(radius_array))

def plot_derivatives(z_c_array):

	dv_dOmegaM_array, dv_domegaDE_array,dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array = calculate_derivatives(z_c_array,'non_flat') 
	dv_dOmegaM_array, dv_dw_array,dv_dh_array, dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array  = calculate_derivatives(z_c_array,'flat') 

	cmap = cm.Spectral_r
	colors = [cmap(i) for i in np.linspace(0, 1, len(z_c_array)-1)]

	fig, ax = plt.subplots(2,2)

	plot_lines_omegaM = [ ax[0,0].plot(radius_array,dv_dOmegaM_array[i], color=color) for i, color in enumerate(colors, start=1)]
	ax[0,0].set_ylabel('$\partial v_{esc}(r,z)/\partial \Omega_m$ [km/s]',fontsize=18)
	ax[0,0].set_xlabel('radius [Mpc]',fontsize=15)

	plot_lines_w = [ ax[0,1].plot(radius_array,dv_dw_array[i], color=color) for i, color in enumerate(colors, start=1)]
	ax[0,1].set_ylabel('$\partial v_{esc}(r,z)/\partial w$ [km/s]',fontsize=18)
	ax[0,1].set_xlabel('radius [Mpc]',fontsize=15)

	plot_lines_h = [ ax[1,0].plot(radius_array,dv_dh_array[i], color=color) for i, color in enumerate(colors, start=1)]
	ax[1,0].set_ylabel('$\partial v_{esc}(r,z)/\partial h$ [km/s]',fontsize=18)
	ax[1,0].set_xlabel('radius [Mpc]',fontsize=15)

	plot_lines_omegaL = [ ax[1,1].plot(radius_array,dv_domegaDE_array[i], color=color) for i, color in enumerate(colors, start=1)]
	# ax[1,1].set_ylabel('$\partial v_{esc}(r,z_n)/\partial h$ [km/s]',fontsize=18)
	ax[1,1].set_ylabel('$\partial v_{esc}(r,z)/\partial \Omega_{\Lambda}$ [km/s]',fontsize=18)
	ax[1,1].set_xlabel('radius [Mpc]',fontsize=15)

	#colorbar
	norm = matplotlib.colors.Normalize(vmin=np.min(z_c_array),vmax=np.max(z_c_array))
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(z_c_array), vmax=np.max(z_c_array)))
	sm._A = [] 	# fake up the array of the scalar mappable. Urgh...


	ax[0,0].set_xlim(0.5,2.5)
	ax[0,0].set_ylim(-100,3000)

	ax[0,1].set_xlim(0.5,2.5)
	ax[0,1].set_ylim(20,170)

	ax[1,0].set_xlim(0.5,2.5)
	# ax[1,0].set_ylim(320,670)

	ax[1,1].set_xlim(0.5,2.5)
	ax[1,1].set_ylim(-1500,100)

	cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])

	plt.colorbar(sm,label='redshift',cax=cbar_ax)

	# plt.xlim(np.min(radius_array),np.max(radius_array))

def plot_derivatives_vary_w(z_c_array):
	dv_dOmegaM_array, dv_dw0_array,dv_dwa_array, dv_dh_array,dv_dbeta_array, dv_dalpha_array, dv_dr_2_array, dv_drho_2_array = calculate_derivatives(z_c_array,'w_z') 

	cmap = cm.Spectral_r
	colors = [cmap(i) for i in np.linspace(0, 1, len(z_c_array)-1)]

	plot_lines_wa = [ plt.plot(radius_array,dv_dwa_array[i], color=color) for i, color in enumerate(colors, start=1)]
	plt.ylabel('$\partial v_{esc}(r,z_n)/\partial w_a$ [km/s]',fontsize=18)
	plt.xlabel('radius [Mpc]',fontsize=15)
	#colorbar
	norm = matplotlib.colors.Normalize(vmin=np.min(z_c_array),vmax=np.max(z_c_array))
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(z_c_array), vmax=np.max(z_c_array)))
	sm._A = [] 	# fake up the array of the scalar mappable. Urgh...

	# cbar_ax = add_axes([0.2, 0.15, 0.01, 0.7])

	plt.colorbar(sm,label='redshift')


	plt.xlim(0.5,2.5)
	plt.ylim(-30,400)

"""""""""""
User input
"""""""""
# cosmo_case = 'w_z' #w_z, non_flat, flat, w_z_riess16_h, non_flat_riess_prior

"""specify redhsift range and Nclus"""
# N_clus = 100 #number of clusters
# redshift_array = np.linspace(0.001 , 0.8, N_clus).round(5) #uniform distribution

"""specify cluster parameter uncertainties"""
# prior_case = '40pct_none' #see cases in 'cluster_uncertainty_params' function
# sigma_squared_list, cluster_edge_unc = cluster_uncertainty_params(prior_case)

"""calculate uncertainties"""
# G_matrix_user_input = make_G_matrix(redshift_array, cosmo_case,sigma_squared_list,cluster_edge_unc)
# plot_2d_contours_from_G(G_matrix_user_input,cosmo_case)





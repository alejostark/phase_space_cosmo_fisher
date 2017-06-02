from __future__ import division
from __future__ import print_function
from math import *
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
from clusterlensing import ClusterEnsemble
from astropy.cosmology import Flatw0waCDM
from astropy.table import Table, vstack


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Likelihood analysis of projected mass density function (\Sigma) """
""" (1) Create a cluster Sigma profile with ClusterEnsemble		
	(2) Sample posterior of likelihood	 """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

prior_list = ['none','fixed']

""""""""""""""""""""""""""""""""""""
""""""""" Fiducial params"""""""""
""""""""""""""""""""""""""""""""""""
omega_M_fid = 0.3
w_fid = -1.
little_h_fid= 0.7

M200_fid = 4e14 #Msun/1e15
log_m200_fid = np.log10(M200_fid)
z_fid = 0.3

result= omega_M_fid, w_fid, little_h_fid, log_m200_fid


""""""""""""""""""""""""""""""""""""
"""""""""init mcmc walkers"""""""""
""""""""""""""""""""""""""""""""""""
ndim = 4 
nwalkers = 40 #Number of MCMC walkers, >> ndim
nburn = 500 # "burn in" period to stabilize search
nsteps = 30000 # number of MCMC steps to take

""""""""""""""""""""""""""""""""""""
""""""""" Functions"""""""""
""""""""""""""""""""""""""""""""""""
#x := r, y:= mass density profile (Sigma), y_err=error on Sigma"

# prior = 'none' #pick whether to put cosmo prior('fixed') or 'none'

def lnlikelihood(theta, x, y, yerr):
	#read model parameters
	omega_M , w , little_h, log_m200 = theta

	M200 = 10.**log_m200
	#calculate model 
	c = ClusterEnsemble([z_fid], cosmology=Flatw0waCDM(H0=little_h*100., Om0=omega_M, w0=w, wa=0.))
	c.m200 = [M200] #calculate profile with this M200, assuming cosmo above and mass-conc relation
	c.calc_nfw(x)    # calculate the profiles
	surface_mass_density_profile = c.sigma_nfw  # access the profiles

	model = surface_mass_density_profile.value

	#error on Sigma 
	inv_sigma2 = 1.0/(yerr**2.)
	
	#return chi2
	return -0.5*(np.sum( (y-model)**2. * inv_sigma2 )) 

def lnprior(theta):
	#assume flatness:
	omega_M , w , little_h, log_m200 = theta

	if prior == 'none':
		if 0 <= omega_M <= 1 \
		and -2. <= w <= 0. \
		and 0.1 <= little_h <= 1.3 \
		and  14. <= log_m200 <= 15.7 : #from 1e14 to 4e15
			return 0.0
		return -np.inf

	elif prior == 'fixed':
		dp = 1e-3
		if omega_M_fid-dp <= omega_M <= omega_M_fid+dp \
		and w_fid-dp <= w <= w_fid+dp \
		and little_h_fid-dp <= little_h <= little_h_fid+dp \
		and  14. <= log_m200 <= 15.7 : #from 1e14 to 4e15
			return 0.0
		return -np.inf

	elif prior == 'fixedh':
		dp = 1e-3
		if 0 <= omega_M <= 1 \
		and -2. <= w <= 0. \
		and little_h_fid-dp <= little_h <= little_h_fid+dp \
		and  14. <= log_m200 <= 15.7 : #from 1e14 to 4e15
			return 0.0
		return -np.inf

def lnposterior(theta, x, y, yerr):
    lnpr = lnprior(theta)
    if not np.isfinite(lnpr):
        return -np.inf
    return lnpr + lnlikelihood(theta, x, y, yerr)

""""""""""""""""""""""""""""""""""""
""""""""" create cluster  """""""""
""""""""""""""""""""""""""""""""""""

#radial bins
rmin, rmax = 0.1, 2. # Mpc
nbins = 11
rbins = np.logspace(np.log10(rmin), np.log10(rmax), num = nbins)

#mass error
perc_error = 0.40

#cluster-making/sigma-making function
def create_cluster_Sigma_profile(z,little_h,omega_M,w,log_m200):
	### create cluster Sigma (mass density) profile for a given z and cosmology ###

	M200= 10.**log_m200
	z = [z]
	c = ClusterEnsemble(z, cosmology=Flatw0waCDM(H0=little_h*100., Om0=omega_M, w0=w, wa=0.))

	#calculate profile
	c.m200 = [M200]
	c.calc_nfw(rbins)
	sigma = c.sigma_nfw

	#calculate M200+ deltaM200 profile with given perc_error 
	c.m200 = [M200 + M200 * perc_error]
	c.calc_nfw(rbins)
	sigma_plus = c.sigma_nfw

	delta_sigma = sigma_plus-sigma

	y = sigma[0].value
	y_err = delta_sigma[0].value

	return y, y_err

#create ten cluster profiles scattered around mean
y_true, yerr_true = create_cluster_Sigma_profile(z_fid,little_h_fid,omega_M_fid,w_fid,log_m200_fid)
x_true = rbins

ten_cluster_profiles_list = []

for _ in range(10):
	radial_cluster_profile = np.zeros_like(rbins)*0.
	for R_bin in range(0,len(x_true)):
		mu, sigma = y_true[R_bin], yerr_true[R_bin] # mean and standard deviation
		radial_cluster_profile[R_bin] = np.random.normal(mu, sigma, 1)

	ten_cluster_profiles_list.append(radial_cluster_profile)

def plot_profiles(x):
	for clus_num in range(0,10):
		plt.scatter(x_true,ten_cluster_profiles_list[clus_num]*(1e6)**2.,color='grey',)

	ytitle = '$\Sigma(r)\ [\mathrm{M}_\mathrm{sun}/\mathrm{Mpc}^2]$'
	plt.ylabel(ytitle, fontsize=20)
	plt.xlabel('$r\ [\mathrm{Mpc}]$', fontsize=20)

	plt.errorbar(x_true, y_true*(1e6)**2., yerr=yerr_true*(1e6)**2.,color='black')
	plt.ylim(1e13,1e16)
	plt.xlim(0,2)
	plt.xscale('log')
	plt.xlim(0,3)

	plt.yscale('log')
	plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""" run MCMC on loop for Nclus profiles  """""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""

for prior_list_index in range(0,len(prior_list)):

	prior = prior_list[prior_list_index]
	for cluster_list_index in range(0,len(ten_cluster_profiles_list)):

		""""""""""""""""""""""""""""""""""""
		""""""""""""""""""""""""""""""""""""
		""""""""" call cluster  """""""""
		""""""""""""""""""""""""""""""""""""
		""""""""""""""""""""""""""""""""""""
		y, yerr = ten_cluster_profiles_list[cluster_list_index],yerr_true
		x = rbins

		""""""""""""""""""""""""""""""""""""
		""""""""""""""""""""""""""""""""""""
		"""sample posterior distribution"""
		""""""""""""""""""""""""""""""""""""
		""""""""""""""""""""""""""""""""""""

		#1e-2 used to be 1e-4
		pos = [result + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, args=(x, y, yerr))

		sampler.run_mcmc(pos, nsteps)
		samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

		""""""""""""""""""""""""""""""""""""
		""""""""""""""""""""""""""""""""""""
		"""				plots 			"""
		""""""""""""""""""""""""""""""""""""
		""""""""""""""""""""""""""""""""""""

		def plot_flat_omegaM_step(x):
			plt.close()

			for i in range(0,nwalkers):
				plt.plot(np.arange(0,nsteps),sampler.chain[i][:,0],color='black',alpha=0.2)
				plt.ylabel('$\Omega_M$',fontsize=20)
				plt.xlabel('step number',fontsize=20)
				plt.axhline(omega_M_fid,color='gray',linewidth=2)

			plt.show()

		def plot_flat_w_step(x):
			plt.close()

			for i in range(0,nwalkers):
				plt.plot(np.arange(0,nsteps),sampler.chain[i][:,1],color='black',alpha=0.2)
				plt.ylabel('$w$',fontsize=20)
				plt.xlabel('step number',fontsize=20)
				plt.axhline(w_fid,color='gray',linewidth=2)
			plt.show()

		def plot_flat_h_step(x):
			plt.close()
			for i in range(0,nwalkers):
				plt.plot(np.arange(0,nsteps),sampler.chain[i][:,2],color='black',alpha=0.2)
				plt.ylabel('$h$',fontsize=20)
				plt.xlabel('step number',fontsize=20)
				plt.axhline(little_h_fid,color='gray',linewidth=2)

		def plot_flat_M200_step(x):
			plt.close()

			for i in range(0,nwalkers):
				plt.plot(np.arange(0,nsteps),sampler.chain[i][:,3],color='black',alpha=0.2)
				plt.ylabel('log($M_{200}$)',fontsize=20)
				plt.xlabel(r'step number',fontsize=20)
				plt.axhline(log_m200_fid,color='gray',linewidth=2)
			plt.show()

		
		plt.close('all')
		plt.figure()
		"""sampler.chain has object with shape (nwalkers,nsteps,ndim)"""
		fig = corner.corner(samples, labels=["$\Omega_M$", "$w$", "$h$",r"log$_{10}(M_{200})$"],truths=[omega_M_fid, w_fid, little_h_fid,log_m200_fid])

		# samples2= np.array(samples)
		# fig = corner.corner(samples2, labels=["$\Omega_M$", "$w$", "$h$",r"log$_{10}(M_{200})$"],truths=[omega_M_fid, w_fid, little_h_fid,log_m200_fid])
		# fig.savefig("/Users/alejo/Desktop/emcee_plots/newround/"+prior_list[prior_list_index]+'_'+str(z_fid)+".png")
		fig.savefig("/Users/alejo/Desktop/emcee_plots/scatter/40pct/"+prior_list[prior_list_index]+'_clus'+str(cluster_list_index)+".png")

		samples_logm200 = samples[:, 3]
		t=Table([samples_logm200],names=['samplesM200'])
		# savename=('/Users/alejo/Desktop/emcee_plots/newround/'+prior_list[prior_list_index]+'_'+str(z_fid)+'.fits')
		savename=('/Users/alejo/Desktop/emcee_plots/scatter/40pct/'+prior_list[prior_list_index]+'_clus'+str(cluster_list_index)+'.fits')

		t.write(savename)


		# plt.hist(np.log(M200_samples),color='grey',bins=20)
		# plt.axvline(log_m200_fid,color='red',ls='--')

		# plus_m200=  (M200_fid + M200_fid * perc_error)/1e15
		# minus_m200=  (M200_fid - M200_fid * perc_error)/1e15

		# plt.axvline(plus_m200,color='red',ls=':')
		# plt.axvline(minus_m200,color='red',ls=':')

		# plt.axvline(log_m200_fid+m200_mcmc[3],color='blue',ls=':')


		# samples[:, 3] = np.exp(samples[:, 3])
		# oM_mcmc, w_mcmc, h_mcmc, m200_mcmc = map(lambda v: (v[1], v[3]-v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 95,99],axis=0)))


		# plt.close('all')
		# plt.hist(samples_30k_40pct,color='red',label='no cosmo prior',histtype='step',bins=30)
		# plt.hist(samples[:, 3],color='black',label='cosmology fixed',histtype='step',bins=30)





 
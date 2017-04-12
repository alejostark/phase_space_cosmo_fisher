from functions_fisher import *
from fisher_matrix import *

import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch


"""produce figures in paper"""

#levels for contour plots
f_1sig = 0.434 #68% CL ellipse
f_2sig = 0.167 #95% CL ellipse
f_3sig = 0.072 #99% CL ellipse

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

""" contours """
def plot_fig5_flat(plot):
	######Prior from Lokas paper on SINGLE beta######
	sigma_beta_squared =  (0.5)**2.  

	####40% mass error#####
	sigma_r2_squared = (0.074)**2.
	sigma_rho2_squared = (11.218e12/1e14)**2.
	sigma_alpha_squared = (0.002)**2.

	######20% mass error######
	sigma_r2_squared_20 = (0.035)**2.
	sigma_rho2_squared_20 = (6.576e12/1e14)**2.
	sigma_alpha_squared_20 = (0.001)**2. 

	sigma_squared_list = [sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared]
	sigma_squared_list_20 = [sigma_beta_squared, sigma_alpha_squared_20, sigma_r2_squared_20, sigma_rho2_squared_20]

	cluster_edge_unc = np.sqrt(50**2 + 50**2 + (1000*0.25/2)**2.) 

	#make G matrix
	G_flat_100= make_G_matrix(np.linspace(0.0001 , 0.8, 100).round(5), 'flat',sigma_squared_list,cluster_edge_unc)
	G_flat_1000 = make_G_matrix(np.linspace(0.0001 , 0.8, 1000).round(5), 'flat',sigma_squared_list,cluster_edge_unc)

	print 'plotting..'
	omega_M_array = np.arange(-1,1.,1e-3)
	w_array = np.arange(-5,2, 1e-3)

	x, y = coord(omega_M_array, w_array)# return coordinate matrices from coordinate vectors
	
	print 'making ellipse 1..'

	#### ellipse 1 #####
	z1 = G_flat_100[0]*x**2 + 2.0*G_flat_100[1]*(x*y) + G_flat_100[3]*(y**2)

	plt.contourf(x+0.3, y-1., z1,  [1/f_2sig,1/f_1sig] , colors='black') #marginalized
	plt.contourf(x+0.3, y-1., z1,  [1/f_1sig,1/f_2sig] , colors='gray') #marginalized

	proxy1 = mpatches.Circle((0.5, 0.5), 0.25, facecolor="black",
            edgecolor="gray", linewidth=3)

	print 'making ellipse 2..'

	#### ellipse 2 #####
	z2 = G_flat_1000[0]*x**2 + 2.0*G_flat_1000[1]*(x*y) + G_flat_1000[3]*(y**2)

	plt.contourf(x+0.3, y-1., z2,  [1/f_2sig, 1/f_1sig] , colors='firebrick') #marginalized
	plt.contourf(x+0.3, y-1., z2, [1/f_1sig,1/f_2sig] , colors='red') #marginalized

	proxy2 = mpatches.Circle((0.5, 0.5), 0.25, facecolor="firebrick",
            edgecolor="red", linewidth=3)

	### plot legend 
	plt.legend([proxy1,proxy2], ['$N_{clus}= 100$' ,'$N_{clus}= 1000$' ],
       handler_map={mpatches.Circle: HandlerEllipse()},frameon=False)


	plt.xlabel('$\Omega_M$',fontsize=20)
	plt.ylabel('$w$',fontsize=20)

	plt.xlim(.26,.34)
	plt.ylim(-3,1)

	plt.show()
	
def plot_fig6_w_z(plot):

	reds_array= np.linspace(0.0001 , 0.8, 100).round(5)

	######Prior from Lokas paper on SINGLE beta######
	sigma_beta_squared =  (0.5)**2.  

	####40% mass error#####
	sigma_r2_squared = (0.074)**2.
	sigma_rho2_squared = (11.218e12/1e14)**2.
	sigma_alpha_squared = (0.002)**2.

	######20% mass error######
	sigma_r2_squared_20 = (0.035)**2.
	sigma_rho2_squared_20 = (6.576e12/1e14)**2.
	sigma_alpha_squared_20 = (0.001)**2. 

	sigma_squared_list = [sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared]
	sigma_squared_list_20 = [sigma_beta_squared, sigma_alpha_squared_20, sigma_r2_squared_20, sigma_rho2_squared_20]

	cluster_edge_unc = np.sqrt(50**2 + 50**2 + (1000*0.25/2)**2.) 

	###
	G_wz_40 = make_G_matrix(reds_array, 'w_z',sigma_squared_list,cluster_edge_unc)
	G_wz_20 = make_G_matrix(reds_array, 'w_z',sigma_squared_list_20,cluster_edge_unc)
	G_wz_riess_prior =  make_G_matrix(reds_array, 'w_z_riess16_h',sigma_squared_list_20,cluster_edge_unc)

	#marginalize over omegaM,h and other parameters to get w0-wa plane
	w0_array = np.arange(-5,2, 1e-2)
	wa_array = np.arange(-5,5, 1e-2)

	x, y = coord(w0_array,wa_array)# return coordinate matrices from coordinate vectors

	####################
	### ellipse 1: 40%
	####################
	
	z_40 = G_wz_40[0]*x**2 + 2.0*G_wz_40[1]*(x*y) + G_wz_40[3]*(y**2)

	plt.contourf(x-1., y+0., z_40,  [1/f_2sig,1/f_1sig] , colors='lightseagreen') #marginalized
	plt.contourf(x-1., y+0., z_40,  [1/f_1sig,1/f_2sig] , colors='turquoise') #marginalized

	proxy1 = mpatches.Circle((0.5, 0.5), 0.25, facecolor="lightseagreen",
            edgecolor="turquoise", linewidth=3)

	####################
	## ellipse 2: 20%
	####################
	
	z_20 = G_wz_20[0]*x**2 + 2.0*G_wz_20[1]*(x*y) + G_wz_20[3]*(y**2)

	plt.contourf(x-1., y+0., z_20,  [1/f_2sig,1/f_1sig] , colors='darkslateblue') #marginalized
	plt.contourf(x-1., y+0., z_20,  [1/f_1sig,1/f_2sig] , colors='mediumslateblue') #marginalized

	proxy2 = mpatches.Circle((0.5, 0.5), 0.25, facecolor="darkslateblue",
            edgecolor="mediumslateblue", linewidth=3)

	####################
	#ellipse 3: 
	####################
	
	z_riess_prior = G_wz_riess_prior[0]*x**2 + 2.0*G_wz_riess_prior[1]*(x*y) + G_wz_riess_prior[3]*(y**2)

	plt.contourf(x-1., y+0., z_riess_prior,  [1/f_2sig,1/f_1sig] , colors='crimson') #marginalized
	plt.contourf(x-1., y+0., z_riess_prior,  [1/f_1sig,1/f_2sig] , colors='pink') #marginalized

	proxy3 = mpatches.Circle((0.5, 0.5), 0.25, facecolor="crimson",
            edgecolor="pink", linewidth=3)

	####################
	plt.legend([proxy1 , proxy2, proxy3], [r'$40\%$ mass scatter' , r'$20\%$ mass scatter',r'$20\%$ mass scatter + Riess et al 2016 $h$ prior'],
      handler_map={mpatches.Circle: HandlerEllipse()},frameon=False,loc='upper left')

	# plt.legend([proxy1 , proxy3], [r'$40\%$ mass scatter'  ,r'$20\%$ mass scatter and $7\%$ $\beta$ scatter'],handler_map={mpatches.Circle: HandlerEllipse()},frameon=False,loc='upper left')

	plt.xlabel('$w_0$',fontsize=20)
	plt.ylabel('$w_a$',fontsize=20)

	plt.xlim(-3,1)
	plt.ylim(-2,2)
	plt.show()

def plot_fig7_nonflat(plot):

	######Prior from Lokas paper on SINGLE beta######
	sigma_beta_squared =  (0.5)**2.  

	####40% mass error#####
	sigma_r2_squared = (0.074)**2.
	sigma_rho2_squared = (11.218e12/1e14)**2.
	sigma_alpha_squared = (0.002)**2.

	sigma_squared_list = [sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared]

	cluster_edge_unc = np.sqrt(50**2 + 50**2 + (1000*0.25/2)**2.) 

	#make G matrix
	G_nonflat = make_G_matrix(np.linspace(0.0001,.8,100).round(5),'non_flat',sigma_squared_list,cluster_edge_unc)
	G_nonflat_riess = make_G_matrix(np.linspace(0.0001,.8,100).round(5),'non_flat_riess_prior',sigma_squared_list,cluster_edge_unc)

	omega_M_array = np.arange(-1,2.,2e-3)
	omega_DE_array = np.arange(-1,2,2e-3)

	x, y = coord(omega_M_array, omega_DE_array)# return coordinate matrices from coordinate vectors

	#### ellipse 1 #####
	z1 = G_nonflat[0]*x**2 + 2.0*G_nonflat[1]*(x*y) + G_nonflat[3]*(y**2)

	plt.contourf(x+0.3, y+0.7 , z1, [0, 1/f_2sig] , colors='gray') #marginalized
	plt.contourf(x+0.3, y+0.7 , z1,  [0,1/f_1sig] , colors='black') #marginalized

	proxy1 = mpatches.Circle((0.5, 0.5), 0.25, facecolor="black",
	        edgecolor="gray", linewidth=3)

	# #### ellipse 2 #####
	z2 = G_nonflat_riess[0]*x**2 + 2.0*G_nonflat_riess[1]*(x*y) + G_nonflat_riess[3]*(y**2)

	plt.contourf(x+0.3, y+0.7 , z2,  [1/f_2sig, 1/f_1sig] , colors='darkgreen') #marginalized
	plt.contourf(x+0.3, y+0.7 , z2, [1/f_1sig,1/f_2sig] , colors='lightgreen') #marginalized

	proxy2 = mpatches.Circle((0.5, 0.5), 0.25, facecolor="darkgreen",
	        edgecolor="lightgreen", linewidth=3)

	plt.legend([proxy1,proxy2], ['no $h$ prior' ,'Riess et al 2016 $h$ prior' ],
	   handler_map={mpatches.Circle: HandlerEllipse()},frameon=False,loc='lower right')

	plt.xlabel('$\Omega_M $',fontsize=20)
	plt.ylabel('$\Omega_{\Lambda}$',fontsize=20)

	plt.xlim(0,.7)
	plt.ylim(0,1.4)

"""other"""
def plot_fig9_sigma_zmin(plot):

	""
	"stacked"
	""
	######Prior from Lokas paper on SINGLE beta######
	sigma_beta_squared_stacked =  (0.02)**2. 

	sigma_r2_squared_stacked = (0.01)**2.
	sigma_rho2_squared_stacked = (1.635e12/1e14)**2.
	sigma_alpha_squared_stacked = (-0.0001)**2. 

	sigma_squared_list_stacked = [sigma_beta_squared_stacked, sigma_alpha_squared_stacked, sigma_r2_squared_stacked, sigma_rho2_squared_stacked]

	cluster_edge_unc_stacked = np.sqrt(50**2 + (1000 * 0.15/2)**2 )

	""
	"NON stacked"
	""
	######Prior from Lokas paper on SINGLE beta######
	sigma_beta_squared =  (0.5)**2.

	####40% mass error#####
	sigma_r2_squared = (0.074)**2.
	sigma_rho2_squared = (11.218e12/1e14)**2.
	sigma_alpha_squared = (0.002)**2.

	######20% mass error######
	sigma_r2_squared_20 = (0.035)**2.
	sigma_rho2_squared_20 = (6.576e12/1e14)**2.
	sigma_alpha_squared_20 = (0.001)**2. 

	sigma_squared_list = [sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared]
	sigma_squared_list_20 = [sigma_beta_squared, sigma_alpha_squared_20, sigma_r2_squared_20, sigma_rho2_squared_20]

	cluster_edge_unc = np.sqrt(50**2 + 50**2 + (1000*0.25/2)**2.) 

	""
	""
	# z_min_array= np.linspace(0.01,.6,7)
	z_min_array= np.linspace(0.001,.6,7)

	sigma_oM_array = np.zeros_like(z_min_array)*0.
	sigma_w_array = np.zeros_like(z_min_array)*0.

	sigma_oM_array_20 = np.zeros_like(z_min_array)*0.
	sigma_w_array_20 = np.zeros_like(z_min_array)*0.

	sigma_oM_array_stacked = np.zeros_like(z_min_array)*0.
	sigma_w_array_stacked = np.zeros_like(z_min_array)*0.

	for z_min_in in range(0,len(z_min_array)):
		z_c_array = np.linspace(z_min_array[z_min_in],0.8,100).round(5)

		print z_min_in, ' out of ', len(z_min_array)

		G_matrix = make_G_matrix(z_c_array, 'flat',sigma_squared_list,cluster_edge_unc)
		G_matrix_20 = make_G_matrix(z_c_array, 'flat',sigma_squared_list_20,cluster_edge_unc)
		G_matrix_stacked = make_G_matrix(z_c_array, 'flat',sigma_squared_list_stacked,cluster_edge_unc_stacked)

		F_inv = np.linalg.inv( (np.array(np.array(G_matrix),dtype='float')))
		F_inv_20 = np.linalg.inv( (np.array(np.array(G_matrix_20),dtype='float')))
		F_inv_stacked = np.linalg.inv( (np.array(np.array(G_matrix_stacked),dtype='float')))

		sigma_oM_array[z_min_in] = np.sqrt(F_inv[0][0])
		sigma_w_array[z_min_in] = np.sqrt(F_inv[1][1])

		sigma_oM_array_20[z_min_in] = sqrt(F_inv_20[0][0])
		sigma_w_array_20[z_min_in] = sqrt(F_inv_20[1][1])

		sigma_oM_array_stacked[z_min_in] = sqrt(F_inv_stacked[0][0])
		sigma_w_array_stacked[z_min_in] = sqrt(F_inv_stacked[1][1])


	fig, ax = plt.subplots(2,1,sharex= True)
	ax[0].plot(z_min_array,sigma_w_array,color='black',label='40% mass scatter',linewidth=2)	
	ax[0].scatter(z_min_array,sigma_w_array,color='black')	

	ax[0].plot(z_min_array,sigma_w_array_20,color='red',label='20% mass scatter',linewidth=2)	
	ax[0].scatter(z_min_array,sigma_w_array_20,color='red')	

	ax[0].plot(z_min_array,sigma_w_array_stacked,color='green',label='stacked',linewidth=2)	
	ax[0].scatter(z_min_array,sigma_w_array_stacked,color='green')	


	ax[1].plot(z_min_array,sigma_oM_array,color='black',linewidth=2)	
	ax[1].scatter(z_min_array,sigma_oM_array,color='black')	

	ax[1].plot(z_min_array,sigma_oM_array_20,color='red',linewidth=2)	
	ax[1].scatter(z_min_array,sigma_oM_array_20,color='red')	

	ax[1].plot(z_min_array,sigma_oM_array_stacked,color='green',linewidth=2)	
	ax[1].scatter(z_min_array,sigma_oM_array_stacked,color='green')	

	ax[0].set_xlim(-.05,.65)
	plt.xlabel('$z_{min}$',fontsize=20)
	ax[0].set_ylabel('$\sigma_w$',fontsize=25)
	ax[1].set_ylabel('$\sigma_{\Omega_M}}$',fontsize=25)

	ax[0].legend(loc='best',frameon=False)


	plt.show()

def plot_fig8_invArea_oM_w(plot):
	##############################	
	##############################	
	##############################	
	##############################	
	##############################	


	######STACKED ######
	sigma_beta_squared_stacked =  (0.02)**2. 
	cluster_edge_unc_stacked = np.sqrt(50**2 + (1000 * 0.15/2)**2 )

	sigma_r2_squared_stacked = (0.01)**2.
	sigma_rho2_squared_stacked = (1.635e12/1e14)**2.
	sigma_alpha_squared_stacked = (0.0001)**2. 
	sigma_squared_list_5_beta = [sigma_beta_squared_stacked, sigma_alpha_squared_stacked, sigma_r2_squared_stacked, sigma_rho2_squared_stacked]

	######UNSTACKED######
	sigma_beta_squared =  (0.5)**2.  
	cluster_edge_unc = np.sqrt(50**2 + 50**2 + (1000*0.25/2)**2.) 

	#20% mass error#
	sigma_r2_squared_20 = (0.035)**2.
	sigma_rho2_squared_20 = (6.576e12/1e14)**2.
	sigma_alpha_squared_20 = (0.001)**2. 
	sigma_squared_list_20 = [sigma_beta_squared, sigma_alpha_squared_20, sigma_r2_squared_20, sigma_rho2_squared_20]

	#40% mass error#
	sigma_r2_squared = (0.074)**2.
	sigma_rho2_squared = (11.218e12/1e14)**2.
	sigma_alpha_squared = (0.002)**2.
	sigma_squared_list = [sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared]


	##############################	
	##############################	
	##############################	
	##############################	
	##############################	

	############
	#### zmax  ##### 
	############
	# z_max_array = np.array([ 0.2,  0.3,  0.4,  0.5,  0.6,  0.67,0.7,  0.8])

	z_max_array = np.arange(.2,.8,1e-1)

	FoM_array_max = np.zeros_like(z_max_array) *0.
	FoM_20pct_array_max = np.zeros_like(z_max_array) *0.
	FoM_5pct_array_max = np.zeros_like(z_max_array) *0.

	for z_max_in in range(0,len(z_max_array)):
		z_c_array_max = np.linspace(0.001,z_max_array[z_max_in],100).round(5)

		print z_max_in, ' out of ', len(z_max_array)
		print z_c_array_max

		G_matrix_5pct_max = make_G_matrix(z_c_array_max, 'flat',sigma_squared_list_5_beta,cluster_edge_unc_stacked)
		G_inv_tot_5pct_max = np.linalg.inv((np.array(np.array(G_matrix_5pct_max),dtype='float')))
		FoM_5pct_array_max[z_max_in] = sqrt(1./det(Matrix(G_inv_tot_5pct_max)))

		G_matrix_20pct_max = make_G_matrix(z_c_array_max, 'flat',sigma_squared_list_20,cluster_edge_unc)
		G_inv_tot_20pct_max = np.linalg.inv((np.array(np.array(G_matrix_20pct_max),dtype='float'))) 
		FoM_20pct_array_max[z_max_in] = sqrt(1./det(Matrix(G_inv_tot_20pct_max)))

		G_matrix_max = make_G_matrix(z_c_array_max, 'flat',sigma_squared_list,cluster_edge_unc)
		G_inv_tot_max = np.linalg.inv((np.array(np.array(G_matrix_max),dtype='float')))    
		FoM_array_max[z_max_in] = sqrt(1./det(Matrix(G_inv_tot_max)))


	############
	#### zmin  ##### 
	############
	z_min_array= np.array([ 0.0001,  0.2,  0.3,  0.4,  0.5,  0.6])

	FoM_array_min = np.zeros_like(z_min_array) *0.
	FoM_20pct_array_min = np.zeros_like(z_min_array) *0.
	FoM_5pct_array_min = np.zeros_like(z_min_array) *0.

	for z_min_in in range(0,len(z_min_array)):
		z_c_array = np.linspace(z_min_array[z_min_in],0.8,100).round(5)

		print z_min_in, ' out of ', len(z_min_array)
		print z_c_array

		G_matrix_5pct_min = make_G_matrix(z_c_array, 'flat',sigma_squared_list_5_beta,cluster_edge_unc_stacked)
		G_inv_tot_5pct_min = np.linalg.inv((np.array(np.array(G_matrix_5pct_min),dtype='float')))       
		FoM_5pct_array_min[z_min_in] = sqrt(1./det(Matrix(G_inv_tot_5pct_min)))

		G_matrix_20pct_min = make_G_matrix(z_c_array, 'flat',sigma_squared_list_20,cluster_edge_unc)
		G_inv_tot_20pct_min = np.linalg.inv((np.array(np.array(G_matrix_20pct_min),dtype='float')))      
		FoM_20pct_array_min[z_min_in] = sqrt(1./det(Matrix(G_inv_tot_20pct_min)))

		G_matrix_min = make_G_matrix(z_c_array, 'flat',sigma_squared_list,cluster_edge_unc)
		G_inv_tot_min = np.linalg.inv((np.array(np.array(G_matrix_min),dtype='float')))     
		FoM_array_min[z_min_in] = sqrt(1./det(Matrix(G_inv_tot_min)))

	############
	#### plots  ##### 
	############
	fig, ax = plt.subplots(1,2,sharey=True)

	ax[0].plot(z_max_array,FoM_array_max,color='black',linewidth=2)
	ax[0].plot(z_max_array,FoM_20pct_array_max,color='red',linewidth=2)
	ax[0].plot(z_max_array,FoM_5pct_array_max,color='green',linewidth=2)


	ax[1].plot(z_min_array,FoM_array_min,color='black',linewidth=2)
	ax[1].plot(z_min_array,FoM_20pct_array_min,color='red',linewidth=2)
	ax[1].plot(z_min_array,FoM_5pct_array_min,color='green',linewidth=2)

	ax[0].set_ylabel('$1/\sqrt{\mathrm{det}[\mathrm{Cov}(\Omega_M,w)]}$',fontsize=20)

	# ax[0].set_ylabel('A^${-1}(\Omega_M,w)$',fontsize=20)

	ax[0].set_xlabel('$z_{max}$',fontsize=20)
	ax[1].set_xlabel('$z_{min}$',fontsize=20)

	ax[0].text(.25,3.6e4,'stacked',color='green',fontsize=20)
	ax[0].text(.25,1.5e4,'20% mass scatter',color='red',fontsize=20)
	ax[0].text(.25,0.5e4,'40% mass scatter',color='black',fontsize=20)

	ax[0].semilogy()
	ax[1].semilogy()

	ax[0].set_ylim(1e-0,2e5)
	ax[1].set_ylim(1e-0,2e5)

	ax[0].set_xlim(0.2,.8)
	ax[1].set_xlim(0.,.6)

def plot_fig1_req(plot):
	plt.plot(redshift_array,r_eq_flat(redshift_array,alpha_fid,rho_2_fid,r_2_fid,0.35,0.7,-1),color='black',ls=':',label=r'flat $\Lambda$CDM, $\Omega_M = 0.35$',linewidth=1.5)
	plt.plot(redshift_array,r_eq_flat(redshift_array,alpha_fid,rho_2_fid,r_2_fid,0.3,0.7,-1),color='black',ls='--',label=r'flat $\Lambda$CDM, $\Omega_M = 0.3$',linewidth=1.5)
	plt.plot(redshift_array,r_eq_flat(redshift_array,alpha_fid,rho_2_fid,r_2_fid,0.25,0.7,-1),color='black',ls='-',label=r'flat $\Lambda$CDM, $\Omega_M = 0.25$',linewidth=1.5)

	plt.legend(loc='lower right',frameon= False)
	plt.ylabel('$r_{eq}$ [Mpc]' ,fontsize=20)
	plt.xlabel('$z$',fontsize=20)
	plt.xlim(0,.8)
	plt.ylim(0,40)

def plot_fig2_qz(plot):
	reds_arr = np.linspace(-.5,1,200)

	plt.plot(reds_arr,q_z_function(reds_arr,[-.9,.3],'flat'),label='flat $w$CDM, $w= -0.9$',color='black',ls=':',linewidth=1.5)
	plt.plot(reds_arr,q_z_function(reds_arr,[-1,.3],'flat'),label='flat $w$CDM, $w= -1$',color='black',linewidth=1.5,ls='--')
	plt.plot(reds_arr,q_z_function(reds_arr,[-1.1,.3],'flat'),label='flat $w$CDM, $w= -1.1$',color='black',ls='-',linewidth=1.5)

	plt.axhline(0,color='grey',ls='-.')
	plt.xlim(0,.8)
	plt.ylim(-.7,.1)

	plt.ylabel('$q(z)$',fontsize=20)
	plt.xlabel('$z$',fontsize=20)
	plt.legend(loc='lower right',frameon= False)

def plot_fig3_deltaV_radius(plot):

	z_c = 0.001
	
	theta_array = radius_array / D_A(z_c,[w_fid, Omega_M_fid, little_h_fid],'flat')

	v_lambda = v_esc_theory_flat(theta_array, z_c,alpha_fid,rho_2_fid,r_2_fid,beta_fid,Omega_M_fid,little_h_fid,-1.)
	v_quintessence = v_esc_theory_flat(theta_array, z_c,alpha_fid,rho_2_fid,r_2_fid,beta_fid,Omega_M_fid,little_h_fid,-.5)
	v_phantom = v_esc_theory_flat(theta_array, z_c,alpha_fid,rho_2_fid,r_2_fid,beta_fid,Omega_M_fid,little_h_fid,-1.5)


	delta_v_quintessence = (v_quintessence-v_lambda) / v_lambda
	delta_v_phantom_DE = (v_phantom-v_lambda) / v_lambda

	plt.plot(radius_array,delta_v_quintessence,linewidth=2,ls= '-',color='red')
	plt.plot(radius_array,delta_v_phantom_DE,linewidth=2,ls= '-',color='black')

	plt.axhline(0,color='grey',linewidth=2)
	plt.text(0.8,.12, r'quintessence ($w=-0.5$)',rotation=11,color='red',fontsize=20)
	plt.text(0.7,.01, r'$\Lambda$ ($w=-1$)',rotation=0,color='grey',fontsize=20)
	plt.text(1.45,-0.035, r'phantom DE ($w=-1.5$)',rotation=-4.5,color='black',fontsize=20)

	plt.xlabel('radius [Mpc]',fontsize=20)
	plt.ylabel('$\Delta v_{esc}(r) /  v_{esc}(r)$',fontsize=20)

	plt.ylim(-.1,.2)

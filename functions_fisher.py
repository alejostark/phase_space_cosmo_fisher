from __future__ import division
from math import *
import math
from astropy import units as u
from sympy import *
from sympy.matrices import *
import numpy as np
import pylab as p
import scipy
import astropy
import scipy.special as ss
import astropy.constants as astroc
import cosmolopy
import matplotlib.pyplot as plt
import scipy.integrate as integrate

######## constants ########
Msun = 1.9891e+30 #kg
c = 299792.458 # km/s


"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"" 		   COSMOLOGY           			   ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""

def z_trans(cosmo_params,case):
    #calculate transition redshift (ie when q transitions from q>0 to q<0)
    redshift_array= np.arange(0,1.4,1e-7)
    if case == 'w_z':
        w0, wa, omega_M = cosmo_params

    elif case == 'flat':
        w, omega_M = cosmo_params
        
    elif case == 'flat':
        w, omega_M = cosmo_params
        
    
    q_z_array = q_z_function(redshift_array,cosmo_params,case)

    return redshift_array[np.abs(np.subtract.outer(q_z_array, 0.)).argmin(0)]

def q_z_function(z_c,cosmo_params,case):
    """ 
    Returns the deceleration parameter as a function of redshift (z_c) for three cases
    corresponding to different sets of cosmological parameters:
    
    I)     'flat' takes in cosmo_params = w, omega_M
    II)     'w_z' takes in cosmo_params = w0, wa, omega_M
    III)'nonflat' takes in cosmo_params= omega_DE, omega_M

    NOTE: 'w_z' assumes the CPL parametrization of dark energy

    """

    if case == 'w_z':
        w0, wa, omega_M = cosmo_params
        
        #assume flatness:
        omega_DE = 1. - omega_M 

        E_z= np.sqrt(omega_M*(1 + z_c)**3. + (omega_DE*(1 + z_c)**(3*(1 + w0 + wa))) * np.exp(-(3*wa*z_c)/(1 + z_c)) )

        omega_M_z = (omega_M * (1+z_c)**3.) / E_z**2.
        omega_DE_z = (omega_DE*(1+z_c)**(3*(1+w0+wa)) *np.exp(-3*wa*z_c/(1+z_c)) ) / E_z**2.

        return (omega_M_z + omega_DE_z*(1 + 3*w0 + (3*wa*z_c/(1+z_c)) ) )/2.

    elif case == 'flat':
        w, omega_M = cosmo_params

        #assume flatness:
        omega_DE = 1. - omega_M

        E_z= np.sqrt( omega_DE * (1+z_c)**(3.+ 3.*w)  + omega_M * (1+z_c)**3. )
        omega_M_z = (omega_M * (1+z_c)**3.) / E_z**2.
        omega_DE_z = (omega_DE*(1+z_c)**(3.+3.*w)) / E_z**2.

        return ( omega_M_z + omega_DE_z*(1. + 3.*w) )/2.

    elif case == 'nonflat':
        omega_DE, omega_M = cosmo_params

        #omega_K != 0
        E_z= np.sqrt( omega_DE  + omega_M * (1+z_c)**3. + (1. - omega_DE - omega_M ) * (1+z_c)**2. )

        omega_M_z = ( omega_M * (1+z_c)**3. ) / E_z**2.
        omega_DE_z = omega_DE / E_z**2.

        return (omega_M_z/2.) - omega_DE_z 
    
    elif case == 'flat_lambda':
        omega_DE = cosmo_params

        #flat,
        omega_M = 1.- omega_DE  

        E_z= np.sqrt( omega_DE  + omega_M * (1+z_c)**3.)

        omega_M_z = ( omega_M * (1+z_c)**3. ) / E_z**2.
        omega_DE_z = omega_DE / E_z**2.

        return (omega_M_z/2.) - omega_DE_z 

def H_z_function(z_c,cosmo_params,case):
    """
    Returns the Hubble parameter as a function of redshift (z_c) for three cases
    corresponding to different sets of cosmological parameters:
    
    I)     'flat' takes in cosmo_params = w, omega_M, little_h
    II)     'w_z' takes in cosmo_params = w0, wa, omega_M, little_h
    III)'nonflat' takes in cosmo_params= omega_DE, omega_M, little_h

    NOTE: 'w_z' assumes the CPL parametrization of dark energy

    """

    if case == 'w_z':
        w0, wa, omega_M, little_h = cosmo_params
        
        #assume flatness:
        omega_DE = 1. - omega_M

        #Using w(z) from Linder, 2003a; Chevallier and Polarski, 2001.
        H0 = little_h * 100
        
        return H0 * np.sqrt(omega_M*(1 + z_c)**3. + (omega_DE*(1 + z_c)**(3*(1 + w0 + wa))) * np.exp(-(3*wa*z_c)/(1 + z_c)) )


    elif case == 'flat':
        w, omega_M, little_h = cosmo_params 

        #assume flatness:
        omega_DE = 1. - omega_M

        E_z= np.sqrt( omega_DE * (1+z_c)**(3.+ 3.*w)  + omega_M * (1+z_c)**3. )
        H0 = little_h * 100

        return H0 * E_z

    elif case == 'nonflat':

        omega_DE, omega_M, little_h = cosmo_params 

        E_z= np.sqrt( omega_DE  + omega_M * (1+z_c)**3. + (1.- omega_DE - omega_M ) * (1+z_c)**2. )
        
        H0 = little_h * 100

        return H0 * E_z


    elif case == 'flat_lambda':

        omega_DE,  little_h = cosmo_params 

        #flat
        omega_M = 1.- omega_DE  

        E_z= np.sqrt( omega_DE  + omega_M * (1+z_c)**3. )
        
        H0 = little_h * 100

        return H0 * E_z

def r_eq(z_c,alpha,rho_2_1e14,r_2,cosmo_params,case):
    #equivalence radius in Mpc for all cosmology cases

    """cosmology"""
    if case == 'non_flat':
        omega_DE, omega_M, little_h = cosmo_params

        H_z = H_z_function(z_c, [omega_DE, omega_M,little_h],'nonflat')
        q_z= q_z_function(z_c,[omega_DE, omega_M],'nonflat')

    elif case == 'flat':
        w, omega_M, little_h = cosmo_params

        H_z = H_z_function(z_c, [w, omega_M,little_h],'flat')
        q_z= q_z_function(z_c,[w, omega_M],'flat')

    elif case == 'w_z':
        w0, wa, omega_M, little_h = cosmo_params

        H_z = H_z_function(z_c, [w0, wa, omega_M, little_h],'w_z')
        q_z= q_z_function(z_c,[w0, wa, omega_M],'w_z')

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -((G_newton*Mtot) / (q_z * H_z**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc

    return r_eq

def D_A(z_c,cosmo_params,case):
    """ 
    Returns angular diameter distance in Mpc for all three cosmological cases
    as a function of redshift (z_c) for three cases corresponding to different 
    sets of cosmological parameters:
    
    I)     'flat' takes in cosmo_params = w, omega_M
    II)     'w_z' takes in cosmo_params = w0, wa, omega_M
    III)'nonflat' takes in cosmo_params= omega_DE, omega_M

    NOTE: 'w_z' assumes the CPL parametrization of dark energy
    """

    if case == 'flat':
        w, omega_M, little_h = cosmo_params 
        omega_DE = 1 - omega_M  #assume flatness:
        H0 = little_h * 100.

        r_z = (c/H0) * ( integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,'flat'), 0 , z_c)[0])

    elif case == 'nonflat':
        omega_DE, omega_M, little_h = cosmo_params 
        omega_K = 1- omega_M - omega_DE
        H0 = little_h * 100.

        if omega_K == 0.:
            print 'WARNING: you picked a flat cosmology! omegaK = 0!'
            r_z = (c/H0) * ( integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,'flat'), 0 , z_c)[0])
        else:
            r_z = (c / (H0*np.sqrt(np.abs(omega_K)))) * np.sin( np.sqrt(np.abs(omega_K))*(integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,'nonflat'), 0 , z_c)[0]))


    elif case == 'w_z':
        w0, wa, omega_M, little_h = cosmo_params
        omega_DE = 1. - omega_M   #assume flatness:
        H0 = little_h * 100.

        r_z = (c/H0) * ( integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,'w_z'), 0 , z_c)[0])


    return r_z/(1.+z_c) 

def rho_crit_z(redshift,w,omega_M,little_h):
    """
    critical density of flat universe as a function of redshift and cosmology
    output units are Msun/Mpc^3
    """
    H0= 100. * little_h #km/s/Mpc
    rho_crit= cosmolopy.density.cosmo_densities(omega_M_0 = omega_M, omega_lambda_0 = (1.-omega_M) , omega_k_0 = 0. , h = little_h)[0] #Msun/Mpc^3
    return rho_crit * (H_z_function(redshift,[w,omega_M,little_h],'flat')/H0)**2.


"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
""    misc.                                ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""

def g_beta(beta):
    return (3.-2.*beta)/(1-beta)

def make_fisher_matrix(single_derivative_list,sigma_array,N_cosmo):
    """
    makes Ndim by Ndim (Ncosmo + 4*Nclus) fisher matrix from derivatives and data uncertainty
    
    INPUT: 1-dimensional list of ith derivatives (dv_esc/dpi); sigma_v_esc_edge [km/s]  
    OUTPUT: Fij matrix 

    NOTE 1: there is no correlation amongst different clusters, as such, the elements of the blocks of these arrays are zero
    NOTE 2: the top 3x3 matrix includes all clusters whereas the rest only sum over a particular cluster. the reason for this is that all other derivs are 0.

    Not the most elegant solution...but it works.
    """

    #read in derivative profiles for all clusters. i.e., for each derivative, the dimensions are: N_clus * len(radius_array)
    if N_cosmo == 2:
        dv_dcosmo1_Nclus, dv_dcosmo2_Nclus, dv_dbeta_Nclus, dv_dalpha_Nclus, dv_dr_2_Nclus, dv_drho_2_Nclus = single_derivative_list
    elif N_cosmo == 3:
        dv_dcosmo1_Nclus, dv_dcosmo2_Nclus, dv_dcosmo3_Nclus, dv_dbeta_Nclus, dv_dalpha_Nclus, dv_dr_2_Nclus, dv_drho_2_Nclus = single_derivative_list
    elif N_cosmo == 4:
        dv_dcosmo1_Nclus, dv_dcosmo2_Nclus, dv_dcosmo3_Nclus,dv_dcosmo4_Nclus, dv_dbeta_Nclus, dv_dalpha_Nclus, dv_dr_2_Nclus, dv_drho_2_Nclus = single_derivative_list

    #number of clusters
    N_clus = len(dv_dcosmo1_Nclus)

    #number of dimensions
    N_dim = 4 * N_clus + N_cosmo
    
    """cosmological params 3x3 matrix """
    #first three by three matrix where we add up over kth radial bin AND nth cluster
    block1_cosmo = (np.arange(0,N_cosmo*N_cosmo).reshape(N_cosmo,N_cosmo))*0.
    for row_1 in range(0,N_cosmo):
        for col_1 in range(0,N_cosmo):
            block1_cosmo[row_1][col_1] = np.sum( ( single_derivative_list[row_1]*single_derivative_list[col_1])/ sigma_array**2. )

    """null cluster 4x4 matrices """
    #cluster params  1...N yield 0. no cross-correlation between clusters.
    block2_null = np.zeros(shape=(4,4)) * 0.0

    """cluster i = j auto-correlation 4x4 matrices"""
    block3_auto_clus_list = []
    for n_clus in range(0,N_clus):
        matrix3_cosmo = np.zeros(shape=(4,4)) * 0.

        for row_matrix_3 in range(0,4):
            for col_matrix_3 in range(0,4):
                row_matrix_3_mod = row_matrix_3 + N_cosmo
                col_matrix_3_mod = col_matrix_3 + N_cosmo

                matrix3_cosmo[row_matrix_3][col_matrix_3] = np.sum( ( single_derivative_list[row_matrix_3_mod][n_clus] * single_derivative_list[col_matrix_3_mod][n_clus])/ sigma_array**2. )

        block3_auto_clus_list.append(matrix3_cosmo) 

    """cluster-cosmo i != j 3x4 matrices"""
    block4_clus_cosmo_list = []

    for n_clus in range(0,N_clus):
        matrix4_cosmo = np.zeros(shape=(N_cosmo,4)) * 0.

        for row_matrix_4 in range(0,N_cosmo):
            for col_matrix_4 in range(0,4):
                col_matrix_4_mod =  col_matrix_4 + N_cosmo
                matrix4_cosmo[row_matrix_4][col_matrix_4] = np.sum( ( single_derivative_list[row_matrix_4][n_clus]*single_derivative_list[col_matrix_4_mod][n_clus])/ sigma_array**2. )

        block4_clus_cosmo_list.append(matrix4_cosmo)    

    """assemble fisher matrix from matrix blocks defined above"""
    fisher_block_matrix = [ [ [] for colz in range(1+N_clus) ] for rowz in range(1+N_clus) ]

    #make block matrix which contains: 1 cosmo matrix + Nclus matrices
    for block_column in range(0,1+N_clus):
        for block_row in range(0,1+N_clus):

            #place cosmo block in Fisher matrix
            if (block_column == 0 ) & (block_row == 0):
                fisher_block_matrix[block_column][block_row] =   block1_cosmo

            #place cosmo-cluster blocks in Fisher matrix along first "row"
            elif (block_row == 0)  & (block_column != 0):
                fisher_block_matrix[block_column][block_row] =   block4_clus_cosmo_list[block_column-1]

            #place cosmo-cluster blocks in Fisher matrix along first "column"
            elif (block_column == 0) & (block_row != 0):
                fisher_block_matrix[block_column][block_row] = block4_clus_cosmo_list[block_row-1].T

            #place cluster-cluster autocorrelation blocks in Fisher matrix along "diagonal"
            elif (block_column == block_row) & (block_column != 0 ) & (block_row != 0):
                fisher_block_matrix[block_column][block_row] = block3_auto_clus_list[block_column-1]

            #place null matrix in all others
            else:
                fisher_block_matrix[block_column][block_row] = block2_null

    # fisher_block_matrix[BLOCK_col][ROW_col][row_of_sub_matrix]

    fisher_matrix= [] #[ [] for rows in range(N_dim) ]

    for block_row in range(0,1+N_clus):
        number_of_rows = len(np.c_[tuple(x[block_row] for x in fisher_block_matrix)])
        for number_of_rows_index in range(0,number_of_rows):
            fisher_matrix.append(np.c_[tuple(x[block_row] for x in fisher_block_matrix)][number_of_rows_index])
    return Matrix(fisher_matrix)

def make_prior_matrix(sigma_squared_list,N_cosmo,N_clus):
    """
    makes Ndim by Ndim (Ncosmo + 4*Nclus) PRIOR fisher matrix

    INPUT: 1-dimensional list of sigma's in Einasto parameters, number of cosmo par., and number of clusters
    OUTPUT: Fij prior matrix 

    Not the most elegant solution...but it works.

    """

    sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared  = sigma_squared_list

    ### Make 4x4 covariance matrix block ###
    covariant_term1 = 0.
    covariant_term2 = 0.

    correlation_coefficient = -0.7
    covariant_term3 = correlation_coefficient * np.sqrt(sigma_rho2_squared *sigma_r2_squared)

    covariant_matrix= Matrix([ \
            [sigma_beta_squared,                   0,               0., 0                ], \
            [                0., sigma_alpha_squared,  covariant_term1, covariant_term2          ], \
            [                0.,     covariant_term1, sigma_r2_squared, covariant_term3              ], \
            [                0.,     covariant_term2,  covariant_term3,sigma_rho2_squared]  \
                    ])

    ### Make 4x4 Fisher (prior) matrix block ###
    identity_matrix = np.identity(4)
    covariant_matrix_inv = covariant_matrix.inv()

    N_dim = 4 * N_clus + N_cosmo
    
    """cosmological params 3x3 (or 4x4) matrix """
    #first matrix where we add up over kth radial bin AND nth cluster
    block1_cosmo = (np.arange(0,N_cosmo*N_cosmo).reshape(N_cosmo,N_cosmo))*0.

    """null cluster 4x4 matrices """
    #cluster params  1...N yield 0. no cross-correlation between clusters.
    block2_null = np.zeros(shape=(4,4)) * 0.0

    """cluster-sosmo i != j 3x4 matrices"""
    block3_null = np.zeros(shape=(N_cosmo,4)) * 0.0
    
    """assemble fisher matrix from matrix blocks defined above"""
    fisher_block_matrix = [ [ [] for colz in range(1+N_clus) ] for rowz in range(1+N_clus) ]

    #make block matrix which contains: 1 cosmo matrix + Nclus matrices
    for block_column in range(0,1+N_clus):
        for block_row in range(0,1+N_clus):

            #place cosmo block in Fisher matrix
            if (block_column == 0 ) & (block_row == 0):
                fisher_block_matrix[block_column][block_row] =   block1_cosmo

            #place cosmo-cluster blocks in Fisher matrix along first "row"
            elif (block_row == 0)  & (block_column != 0):
                fisher_block_matrix[block_column][block_row] =   block3_null

            #place cosmo-cluster blocks in Fisher matrix along first "column"
            elif (block_column == 0) & (block_row != 0):
                fisher_block_matrix[block_column][block_row] = block3_null.T

            #place cluster-cluster autocorrelation blocks in Fisher matrix along "diagonal"
            elif (block_column == block_row) & (block_column != 0 ) & (block_row != 0):
                fisher_block_matrix[block_column][block_row] = covariant_matrix_inv

            #place null matrix in all others
            else:
                fisher_block_matrix[block_column][block_row] = block2_null

    fisher_matrix= [] #[ [] for rows in range(N_dim) ]

    for block_row in range(0,1+N_clus):
        number_of_rows = len(np.c_[tuple(x[block_row] for x in fisher_block_matrix)])
        for number_of_rows_index in range(0,number_of_rows):
            fisher_matrix.append(np.c_[tuple(x[block_row] for x in fisher_block_matrix)][number_of_rows_index])
    return Matrix(fisher_matrix)    

def make_prior_matrix_cosmo_prior(sigma_squared_list,N_cosmo,N_clus):
    """
    ******** Adds Riess cosmological prior on little_h **********

    makes Ndim by Ndim (Ncosmo + 4*Nclus) PRIOR fisher matrix

    INPUT: 1-dimensional list of sigma's in Einasto parameters, number of cosmo par., and number of clusters
    OUTPUT: Fij prior matrix 

    Not the most elegant solution...but it works.

    """

    sigma_beta_squared, sigma_alpha_squared, sigma_r2_squared, sigma_rho2_squared  = sigma_squared_list

    ### Make 4x4 covariance matrix block ###
    covariant_term1 = 0.
    covariant_term2 = 0.

    correlation_coefficient = -0.7
    covariant_term3 = correlation_coefficient * np.sqrt(sigma_rho2_squared *sigma_r2_squared)

    covariant_matrix= Matrix([ \
            [sigma_beta_squared,                   0,               0., 0                ], \
            [                0., sigma_alpha_squared,  covariant_term1, covariant_term2          ], \
            [                0.,     covariant_term1, sigma_r2_squared, covariant_term3              ], \
            [                0.,     covariant_term2,  covariant_term3,sigma_rho2_squared]  \
                    ])

    ### Make 4x4 Fisher (prior) matrix block ###
    identity_matrix = np.identity(4)
    # covariant_matrix_inv = Matrix(np.linalg.solve(covariant_matrix, identity_matrix))
    covariant_matrix_inv = covariant_matrix.inv()

    N_dim = 4 * N_clus + N_cosmo
    
    """cosmological params 3x3 matrix w/ riess prior"""
    sigma_h_squared = (1.74/100)**2. #From riess et al 2016

    if N_cosmo == 3:    
        block1_cosmo_matrix= Matrix([ \
            [ 0.,  0.,  0.                ], \
            [ 0., 0.,  0.                ], \
            [ 0., 0., 1./sigma_h_squared  ]])

    elif N_cosmo == 4:
        block1_cosmo_matrix= Matrix([ \
            [ 0., 0.,  0.,  0.                  ], \
            [ 0., 0.,  0.,  0.                  ], \
            [ 0., 0.,  0.,  0.                  ], \
            [ 0., 0.,  0.,  1./sigma_h_squared  ]])

    block1_cosmo  =  np.array(block1_cosmo_matrix)

    """null cluster 4x4 matrices """
    #cluster params  1...N yield 0. no cross-correlation between clusters.
    block2_null = np.zeros(shape=(4,4)) * 0.0

    """cluster-sosmo i != j 3x4 matrices"""
    block3_null = np.zeros(shape=(N_cosmo,4)) * 0.0
    
    """assemble fisher matrix from matrix blocks defined above"""
    fisher_block_matrix = [ [ [] for colz in range(1+N_clus) ] for rowz in range(1+N_clus) ]

    #make block matrix which contains: 1 cosmo matrix + Nclus matrices
    for block_column in range(0,1+N_clus):
        for block_row in range(0,1+N_clus):

            #place cosmo block in Fisher matrix
            if (block_column == 0 ) & (block_row == 0):
                fisher_block_matrix[block_column][block_row] =   block1_cosmo

            #place cosmo-cluster blocks in Fisher matrix along first "row"
            elif (block_row == 0)  & (block_column != 0):
                fisher_block_matrix[block_column][block_row] =   block3_null

            #place cosmo-cluster blocks in Fisher matrix along first "column"
            elif (block_column == 0) & (block_row != 0):
                fisher_block_matrix[block_column][block_row] = block3_null.T

            #place cluster-cluster autocorrelation blocks in Fisher matrix along "diagonal"
            elif (block_column == block_row) & (block_column != 0 ) & (block_row != 0):
                fisher_block_matrix[block_column][block_row] = covariant_matrix_inv

            #place null matrix in all others
            else:
                fisher_block_matrix[block_column][block_row] = block2_null

    fisher_matrix= [] #[ [] for rows in range(N_dim) ]

    for block_row in range(0,1+N_clus):
        number_of_rows = len(np.c_[tuple(x[block_row] for x in fisher_block_matrix)])
        for number_of_rows_index in range(0,number_of_rows):
            fisher_matrix.append(np.c_[tuple(x[block_row] for x in fisher_block_matrix)][number_of_rows_index])
    return Matrix(fisher_matrix)    

def angsep(ra1deg,dec1deg,ra2deg,dec2deg):
    """ Determine separation in degrees between two celestial objects 
        arguments are RA and Dec in decimal degrees. 
    """
    ra1rad=ra1deg*np.pi/180.
    dec1rad=dec1deg*np.pi/180.
    ra2rad=ra2deg*np.pi/180.
    dec2rad=dec2deg*np.pi/180.
    
    # calculate scalar product for determination
    # of angular separation
    
    x=np.cos(ra1rad)*np.cos(dec1rad)*np.cos(ra2rad)*np.cos(dec2rad)
    y=np.sin(ra1rad)*np.cos(dec1rad)*np.sin(ra2rad)*np.cos(dec2rad)
    z=np.sin(dec1rad)*np.sin(dec2rad)
    
    rad=np.arccos(x+y+z) # Sometimes gives warnings when coords match
    
    # use Pythargoras approximation if rad < 1 arcsec
    sep = np.choose( rad<0.000004848 , (
        np.sqrt((np.cos(dec1rad)*(ra1rad-ra2rad))**2+(dec1rad-dec2rad)**2),rad))
        
    # Angular separation
    sep=sep*180./np.pi

    return sep


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))

def concentration_meta(mass,redshift,little_h):
    """
    input m200 & cosmology --> c200
    (concentration relation used in Sereno's metacatalog)

    NOTE: input masses must be in same cosmology as little_h listed
    and in units of Msun
    """
    A = 5.71 
    B = -0.084 
    C= -0.47 
    Mpivot = 2e12/little_h

    c200 = A * (mass/Mpivot)**B * (1+redshift)**C
    
    return c200



"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"" 		    ESCAPE VELOCITY PROFILES	   ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""


def v_esc_theory_w_z(theta,z_c,alpha,rho_2_1e14,r_2,beta,omega_M,little_h,w0,wa):
    """cosmology"""
    H_z = H_z_function(z_c, [w0, wa, omega_M, little_h],'w_z')
    q_z= q_z_function(z_c,[w0, wa, omega_M],'w_z')

    r = theta * D_A(z_c,[w0, wa, omega_M, little_h],'w_z')

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -((G_newton*Mtot) / (q_z * H_z**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc

    """if r_eq -> infinity then return vanilla Einasto"""
    # if math.isnan(r_eq)==False:
    if q_z < 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - q_z * (H_z**2.) * (r**2 - r_eq**2) )

    # elif math.isnan(r_eq)==True:
    elif q_z >= 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.*( phi_ein_orig  )) 

    v_esc_projected = v_esc / np.sqrt(g_beta(beta))

    return v_esc_projected

def v_esc_theory_flat(theta,z_c,alpha,rho_2_1e14,r_2,beta,omega_M,little_h,w):
    """cosmology"""
    H_z = H_z_function(z_c, [w, omega_M,little_h],'flat')
    q_z= q_z_function(z_c,[w, omega_M],'flat')

    r = theta * D_A(z_c,[w, omega_M, little_h],'flat')

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -((G_newton*Mtot) / (q_z * H_z**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc

    """if r_eq -> infinity then return vanilla Einasto"""
    # if math.isnan(r_eq)==False:
    if q_z < 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - q_z * (H_z**2.) * (r**2 - r_eq**2) )

    # elif math.isnan(r_eq)==True:
    elif q_z >= 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.*( phi_ein_orig  )) 

    v_esc_projected = v_esc / np.sqrt(g_beta(beta))

    return v_esc_projected

def v_esc_theory_non_flat(theta,z_c,alpha,rho_2_1e14,r_2,beta,omega_M,little_h,omega_DE):
    """cosmology"""
    H_z = H_z_function(z_c, [omega_DE, omega_M,little_h],'nonflat')
    q_z= q_z_function(z_c,[omega_DE, omega_M],'nonflat')

    r = theta * D_A(z_c,[omega_DE, omega_M, little_h],'nonflat')

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -((G_newton*Mtot) / (q_z * H_z**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc


    """if r_eq -> infinity then return vanilla Einasto"""
    if q_z < 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - q_z * (H_z**2.) * (r**2 - r_eq**2) )

    elif q_z >= 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.*( phi_ein_orig  )) 

    v_esc_projected = v_esc / np.sqrt(g_beta(beta))
    return v_esc_projected



def v_esc_theory_non_flat_lil(theta,z_c,alpha,rho_2_1e14,r_2,beta,lil_omega_M,lil_omega_lambda):
    """cosmology"""
    qHsquared = ( (0.5 * lil_omega_M * (1+z_c)**3.)- lil_omega_lambda) * 100.**2.

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -(G_newton*Mtot) / qHsquared #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc


    """if r_eq -> infinity then return vanilla Einasto"""
    if math.isnan(r_eq) == False:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - qHsquared * (r**2 - r_eq**2) )

    elif math.isnan(r_eq) == True:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.* phi_ein_orig  ) 

    v_esc_projected = v_esc / np.sqrt(g_beta(beta))
    return v_esc_projected

def v_esc_theory_flat_lambda(theta,z_c,alpha,rho_2_1e14,r_2,beta,little_h,omega_DE):
    """cosmology"""
    H_z = H_z_function(z_c, [omega_DE,little_h],'flat_lambda')
    q_z= q_z_function(z_c,omega_DE,'flat_lambda')

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -((G_newton*Mtot) / (q_z * H_z**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc

    """if r_eq -> infinity then return vanilla Einasto"""
    if q_z < 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - q_z * (H_z**2.) * (r**2 - r_eq**2) )

    elif q_z >= 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.*( phi_ein_orig  )) 

    v_esc_projected = v_esc / np.sqrt(g_beta(beta))
    return v_esc_projected



"""" OLD ... w/o angular diameter"""
def v_esc_theory_w_z_old(r,z_c,alpha,rho_2_1e14,r_2,beta,omega_M,little_h,w0,wa):
    """cosmology"""
    
    H_z = H_z_function(z_c, [w0, wa, omega_M, little_h],'w_z')
    q_z= q_z_function(z_c,[w0, wa, omega_M],'w_z')

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -((G_newton*Mtot) / (q_z * H_z**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc

    """if r_eq -> infinity then return vanilla Einasto"""
    # if math.isnan(r_eq)==False:
    if q_z < 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - q_z * (H_z**2.) * (r**2 - r_eq**2) )

    # elif math.isnan(r_eq)==True:
    elif q_z >= 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.*( phi_ein_orig  )) 

    v_esc_projected = v_esc / np.sqrt(g_beta(beta))

    return v_esc_projected

def v_esc_theory_flat_old(r,z_c,alpha,rho_2_1e14,r_2,beta,omega_M,little_h,w):
    """cosmology"""
    H_z = H_z_function(z_c, [w, omega_M,little_h],'flat')
    q_z= q_z_function(z_c,[w, omega_M],'flat')

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -((G_newton*Mtot) / (q_z * H_z**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc

    """if r_eq -> infinity then return vanilla Einasto"""
    # if math.isnan(r_eq)==False:
    if q_z < 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - q_z * (H_z**2.) * (r**2 - r_eq**2) )

    # elif math.isnan(r_eq)==True:
    elif q_z >= 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.*( phi_ein_orig  )) 

    v_esc_projected = v_esc / np.sqrt(g_beta(beta))

    return v_esc_projected

def v_esc_theory_non_flat_old(r,z_c,alpha,rho_2_1e14,r_2,beta,omega_M,little_h,omega_DE):
    """cosmology"""
    H_z = H_z_function(z_c, [omega_DE, omega_M,little_h],'nonflat')
    q_z= q_z_function(z_c,[omega_DE, omega_M],'nonflat')


    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -((G_newton*Mtot) / (q_z * H_z**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc


    """if r_eq -> infinity then return vanilla Einasto"""
    if q_z < 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - q_z * (H_z**2.) * (r**2 - r_eq**2) )

    elif q_z >= 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.*( phi_ein_orig  )) 

    v_esc_projected = v_esc / np.sqrt(g_beta(beta))
    return v_esc_projected

def v_esc_theory_non_flat_lil_old(r,z_c,alpha,rho_2_1e14,r_2,beta,lil_omega_M,lil_omega_lambda):
    """cosmology"""
    qHsquared = ( (0.5 * lil_omega_M * (1+z_c)**3.)- lil_omega_lambda) * 100.**2.

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -(G_newton*Mtot) / qHsquared #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc


    """if r_eq -> infinity then return vanilla Einasto"""
    if math.isnan(r_eq) == False:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - qHsquared * (r**2 - r_eq**2) )

    elif math.isnan(r_eq) == True:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.* phi_ein_orig  ) 

    v_esc_projected = v_esc / np.sqrt(g_beta(beta))
    return v_esc_projected

def v_esc_theory_flat_lambda_old(r,z_c,alpha,rho_2_1e14,r_2,beta,little_h,omega_DE):
    """cosmology"""
    H_z = H_z_function(z_c, [omega_DE,little_h],'flat_lambda')
    q_z= q_z_function(z_c,omega_DE,'flat_lambda')

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -((G_newton*Mtot) / (q_z * H_z**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc

    """if r_eq -> infinity then return vanilla Einasto"""
    if q_z < 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - q_z * (H_z**2.) * (r**2 - r_eq**2) )

    elif q_z >= 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.*( phi_ein_orig  )) 

    v_esc_projected = v_esc / np.sqrt(g_beta(beta))
    return v_esc_projected


""" un projected Psi """

def phi_w_z(r,z_c,alpha,rho_2_1e14,r_2,omega_M,little_h,w):
    """cosmology"""
    H_z = H_z_function(z_c, [w, omega_M,little_h],'flat')
    q_z= q_z_function(z_c,[w, omega_M],'flat')

    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    r_eq_cubed = -((G_newton*Mtot) / (q_z * H_z**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc

    """if r_eq -> infinity then return vanilla Einasto"""
    # if math.isnan(r_eq)==False:
    if q_z < 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        ### phi r_eq ###
        s_req = r_eq/h

        part1_req = 1. - ( ( ss.gammaincc(3.*n,s_req**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_req = (s_req *  ss.gammaincc(2.*n,s_req**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_req =  -(G_newton* Mtot/(s_req*h)) * (part1_req+part2_req)

        v_esc = np.sqrt(-2.*( phi_ein_orig - phi_ein_req ) - q_z * (H_z**2.) * (r**2 - r_eq**2) )

    # elif math.isnan(r_eq)==True:
    elif q_z >= 0.:
        ### phi orig ###
        s_orig = r/h

        part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
        part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

        phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

        v_esc = np.sqrt(-2.*( phi_ein_orig  )) 

    psi = - 0.5 * v_esc**2.

    return psi


def psi_einasto(r,alpha,rho_2_1e14,r_2):
    """Einasto"""
    ### map betweenEinasto params and Miller et al params###
    rho_2 = rho_2_1e14*1e14

    n = 1/alpha
    rho_0 = rho_2 * np.exp(2.*n)
    h = r_2 / (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
    Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg

    ### phi orig ###
    s_orig = r/h

    part1_orig = 1. - ( ( ss.gammaincc(3.*n,s_orig**(1./n)) * ss.gamma(3.*n) ) / gamma_3n )
    part2_orig = (s_orig *  ss.gammaincc(2.*n,s_orig**(1./n)) * ss.gamma(2.*n) ) / gamma_3n

    phi_ein_orig =  -(G_newton* Mtot/(s_orig*h)) * (part1_orig+part2_orig)

    return phi_ein_orig



"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"" 			   MASS/DENSITY PROFILES	   ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""

###NFW###
def rho_nfw(r,m200,r200,c200):
    """
    Radial rho NFW 
    INPUT: r is radial in [Mpc]
    OUTPUT:  Msun/Mpc^3
    """
    g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
    rho_s = (m200/(4.*np.pi*r200**3.)) * c200**3. * g
    r_s = r200/c200 #scale radius

    return rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )
    
def M_cumulative_nfw(r,m200,r200,c200):
    g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
    term = c200 * (r/r200)
    return g * m200 * (np.log(1+term) - (term/(1+term) )   )

###Einasto###
def rho_einasto(r,alpha,r_2,rho_2):    
    return rho_2 * np.exp( (-2./alpha) * ( (r/r_2)**(alpha) -1 ) )


def rho_einasto2(r,alpha,M,R):    
    n = 1/alpha
    r_2 = R * (2.*n)**n

    d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
    gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))

    rho_0 = M / 4. * np.pi   * (R**3.) * n * gamma_3n #kg
    rho_2 = rho_0 / np.exp(2.*n)

    return rho_2 * np.exp( (-2./alpha) * ( (r/r_2)**(alpha) -1 ) )

    
def rho_einasto_nfw(r,rho_s,r_s):
    alpha = 0.3
    r_2 = r_s 
    rho_2 = rho_s / 4.
    return rho_2 * np.exp( (-2./alpha) * ( (r/r_2)**(alpha) -1 ) )

def rho_einasto_montenegro(r,rho_0,h,n):
    return rho_0 * np.exp( -(r/h)**(1/n)  )






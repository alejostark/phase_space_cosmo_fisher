# phase_space_cosmo_fisher
Codes used to produce Fisher matrix 2D contours found in Stark, Miller Huterer 2016.

----------------------------
MAIN codes

*fisher_matrix.py* 
This is the main code which calculates the fisher matrix and generates 2d marginalized contours of cosmological parameters given a specified redshift array and cosmological case. See comments in code. You can also plot the derivatives used in the fisher matrix through the plot_derivatives function.

*functions_fisher.py* 
Functions such as the escape velocity profile and cosmological quantities used by fisher_matrix.py are housed here.

*paper_figures.py *
Functions to generate the exact plots found in the Stark, Miller, Huterer 2016 paper are stored here.

----------------------------
OTHER codes

*3dplot_param.py*
generates 3D plots of qH^2 and other cosmological quantities as a function of redshift and cosmology

*error_uncertainty_plot.py*
maps NFW errors to Einasto parameter errors 

----------------------------
TUTORIAL

Open fisher_matrix.py to specify the number of clusters and redshift range (see redshift_array) used to generate 2D contours for a given cosmological case. Run to generate 2d contours of whatever cosmological case you specify. Read in-code documentation and see Stark et al 2016 to see the various cases.

----------------------------
LIBRARIES 
you will need cosmolopy, astropy and sympy

----------------------------
If you have any questions email the author: alejo@umich.edu

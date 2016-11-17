from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

from functions_fisher import *


redshift = 0.6


def qHsquared_function(z_c,oDE,oM,little_h):
	return (1 /q_z_function(z_c,[oDE, oM],'nonflat') *  (H_z_function(z_c, [oDE, oM,little_h],'nonflat'))**2.)**(1./3.)

def r_eq_function_nf(z_c,oM,oDE,little_h):
	H_z = H_z_function(z_c, [oDE, oM,little_h],'nonflat')
	q_z= q_z_function(z_c,[oDE, oM],'nonflat')
	alpha = 0.253
	r_2 = 0.563
	rho_2 = 1.358e14

	n = 1/alpha
	rho_0 = rho_2 * np.exp(2.*n)
	h = r_2 / (2.*n)**n

	d_n = (3*n) - (1./3.) + (8./1215.*n) + (184./229635.*n**2.) + (1048./31000725.*n**3.) - (17557576./1242974068875. * n**4.) 
	gamma_3n = 2 * ((ss.gammainc(3*n , d_n) ) * ss.gamma(3*n))
	Mtot = 4. * np.pi * rho_0 * Msun * (h**3.) * n * gamma_3n #kg
	G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.kg).value #Mpc km2/s^2 kg
	req=  (-((G_newton*Mtot) / (q_z * H_z**2.)))**(1./3.) 

	return req

fig = plt.figure()
ax = fig.gca(projection='3d')

N=100

x = np.linspace(0,1, N)
y = np.linspace(0,1 , N)
X, Y = np.meshgrid(x, y)

plt.title('$z=$'+str(redshift))
Z1 = r_eq_function_nf(redshift,X,Y,0.3)
Z2 = r_eq_function_nf(redshift,X,Y,0.7)

ax.plot_surface(X, Y, Z1,color='black',alpha=0.1)
ax.plot_surface(X, Y, Z2,color='purple',alpha=0.1)




ax.set_xlabel('$\Omega_M$')
ax.set_xlim(0, 1)
ax.set_ylabel('$\Omega_{\Lambda}$')
ax.set_ylim(0, 1)

ax.set_zlabel('$r_{eq}$')
ax.set_zlim(0,30)


plt.show()





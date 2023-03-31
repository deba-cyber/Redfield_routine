import sys
import math
import copy
import numpy as np
from numpy import linalg as LA
from scipy.stats import linregress
import matplotlib.pyplot as plt


#######################################
## *** Parameters for LYH paper *** ##
## all parameters start with gate ##
#######################################

# N_site_fig5 = 8  # No of bridge site when to run just for a single N
# N_bridge = [10]  # no of bridge sites
N_bridge = np.arange(1, 11)  # Bridge sites to consider
#N_bridge_fig4 = [3, 4, 5, 6, 7]
#N_bridge_fig6 = [ 6,7,8]
k_B = 1.38064852e-23  # Boltzmann constant in J K**-1
c = 29979245800  # speed of light cm s^-1
h_J = 6.626070040e-34  # Planck constant in J-s
wn_J = 1.9863024582479222e-23   # waveno to Joule conversion factor
wn_eV = 0.00012398419843856836  # waveno to eV conversion factor
J_eV = 6.242e+18  # Joule to eV conversion factor
h_eV = h_J*J_eV		# Planck constant in eV-s
k_B_eV = k_B*J_eV  # Boltzmann constant in eV K**-1


gate_epsilon = 3000  # in cm**-1
gate_epsilon_J = gate_epsilon*wn_J  # in J
gate_epsilon_eV = gate_epsilon*wn_eV  # in eV
gate_D = gate_A = 0  # in eV and J
gate_sd = 0.05  # source-drain voltage in eV
gate_V_M = gate_V_D = gate_V_A = 300  # in cm**-1
gate_V_M_J = gate_V_D_J = gate_V_A_J = 300*wn_J  # in cm**-1
gate_V_M_eV = gate_V_D_eV = gate_V_A_eV = 300*wn_eV  # in eV
# gate_Gamma_A = 0.05     ## in eV
gate_T = 300  # temperature in K
gate_kappa = 100  # dephasing/relaxation rate in cm**-1
# gate_kappa = 30000        ## dephasing/relaxation rate in cm**-1
gate_kappa_J = gate_kappa*wn_J  # dephasing/relaxation rate in J
gate_kappa_eV = gate_kappa*wn_eV  # in eV
gate_tau_c_inv = 600  # in cm**-1
gate_tau_c_inv_J = 600*wn_J  # in J
gate_tau_c_inv_eV = 600*wn_eV  # in eV
# gate_Vg = [-0.15+j*0.01 for j in range(31)]           ## gate voltage in eV
gate_Vg = [0.]  # gate voltage in eV
gate_sd_status = "off"  # source-drain status
gate_voltage_status = "on"  # gate-voltage staus
# gate_J = 200*wn_J       ## J is chosen as 200 cm**-1
gate_J = 200*wn_eV  # J is chosen as 200 cm**-1 in eV unit
# gate_Gamma_a = 400*wn_J ## in J
gate_Gamma_a = 400*wn_eV  # in eV

## parameters in eV unit ##


## ------------------------------- ##
## ------------------------------- ##
gate_Gamma_a = 0.05
gate_T = 300
gate_T_inv = [3.1+0.01*j for j in range(51)]
gate_T_arr = [1000/elem for elem in gate_T_inv]

gate_T_inv_fit = [1/elem for elem in gate_T_arr]

gate_kappa = 0.01
gate_tau_c_inv = 0.0744
gate_sd = 0.05
# array for checking source-drain voltage variation for a given gate voltage
gate_sd_arr = [0.+j*0.008 for j in range(26)]
gate_Vg = [0.]
# array for checking effect of variation of gate voltage
gate_Vg_arr = [-0.15+j*0.01 for j in range(31)]
gate_sd_status = "on"  # source-drain status
gate_voltage_status = "on"  # gate voltage status
gate_J = 200*wn_eV  # J is chosen as 200 cm**-1

######################################
######################################



#######################################################
## Molecular Hamiltonian for the gate control paper ##
#######################################################

## This is for all N, all gate_Vg for a given SD voltage ##

eigval_store = []
eigvec_store = []

for ele in N_bridge:
    eigval_store_N = []
    eigvec_store_N = []
    for elem in gate_Vg:
        gate_H_M = np.zeros((ele+2, ele+2))
        for i in range(1, len(gate_H_M)-1):
            if gate_sd_status == "off":
                gate_H_M[i][i] = gate_epsilon_eV-elem
            elif gate_sd_status == "on":
                gate_H_M[i][i] = gate_epsilon_eV - elem - gate_sd*(i/(ele+1))
            gate_H_M[i][i+1] = gate_H_M[i][i-1] = gate_V_M_eV
#        gate_H_M[len(gate_H_M)-1][len(gate_H_M)-1] = -gate_sd
        gate_H_M[0][1] = gate_H_M[1][0] = gate_V_D*wn_eV
        gate_H_M[len(gate_H_M)-1][len(gate_H_M) -
                                  2] = gate_H_M[len(gate_H_M)-2][len(gate_H_M)-1] = gate_V_A*wn_eV
        AA, BB = LA.eigh(np.real(gate_H_M))
        eigval_store_N.append(AA.tolist())
        eigvec_store_N.append(BB.tolist())
    eigval_store.append(eigval_store_N)
    eigvec_store.append(eigvec_store_N)



## This is for figure 6 ##

"""

eigval_store_fig6 = []
eigvec_store_fig6 = []

for ele in N_bridge_fig6:
    eigval_store_fig6_N = []
    eigvec_store_fig6_N = []
    for elem in gate_sd_arr:
        gate_H_M_fig6 = np.zeros((ele+2, ele+2))
        for i in range(1, len(gate_H_M_fig6)-1):
            gate_H_M_fig6[i][i] = gate_epsilon - gate_Vg[0] - elem*(i/(ele+1))
            gate_H_M_fig6[i][i+1] = gate_H_M_fig6[i][i-1] = gate_V_M
        gate_H_M_fig6[len(gate_H_M_fig6)-1][len(gate_H_M_fig6)-1] = -elem
        gate_H_M_fig6[0][1] = gate_H_M_fig6[1][0] = gate_V_D
        gate_H_M_fig6[len(gate_H_M_fig6)-1][len(gate_H_M_fig6) -
                                            2] = gate_H_M_fig6[len(gate_H_M_fig6)-2][len(gate_H_M_fig6)-1] = gate_V_A
        AA_fig6, BB_fig6 = LA.eigh(np.real(gate_H_M_fig6))
        eigval_store_fig6_N.append(AA_fig6.tolist())
        eigvec_store_fig6_N.append(BB_fig6.tolist())
    eigval_store_fig6.append(eigval_store_fig6_N)
    eigvec_store_fig6.append(eigvec_store_fig6_N)

"""


"""


## This is for figure 4 ##

eigval_store_fig4 = []
eigvec_store_fig4 = []

for ele in N_bridge_fig4:
    eigval_store_fig4_N = []
    eigvec_store_fig4_N = []
    for elem in gate_Vg_arr:
        gate_H_M_fig4 = np.zeros((ele+2, ele+2))
        for i in range(1, len(gate_H_M_fig4)-1):
            if gate_sd_status == "off":
                gate_H_M_fig4[i][i] = gate_epsilon - elem
            elif gate_sd_status == "on":
                gate_H_M_fig4[i][i] = gate_epsilon - elem - gate_sd*(i/(ele+1))
            gate_H_M_fig4[i][i+1] = gate_H_M_fig4[i][i-1] = gate_V_M
        gate_H_M_fig4[len(gate_H_M_fig4)-1][len(gate_H_M_fig4)-1] = - gate_sd
        gate_H_M_fig4[0][1] = gate_H_M_fig4[1][0] = gate_V_D
        gate_H_M_fig4[len(gate_H_M_fig4)-1][len(gate_H_M_fig4) -
                                            2] = gate_H_M_fig4[len(gate_H_M_fig4)-2][len(gate_H_M_fig4)-1] = gate_V_A
        AA_fig4, BB_fig4 = LA.eigh(np.real(gate_H_M_fig4))
        eigval_store_fig4_N.append(AA_fig4.tolist())
        eigvec_store_fig4_N.append(BB_fig4.tolist())
    eigval_store_fig4.append(eigval_store_fig4_N)
    eigvec_store_fig4.append(eigvec_store_fig4_N)

"""

## This is for figure 5 ##


"""
eigval_store_fig5 = []
eigvec_store_fig5 = []

for k in range(len(gate_V_A_arr)):
	eigval_store_fig5_case = []
	eigvec_store_fig5_case = []
	for j in range(len(gate_V_A_arr[k])):
		gate_H_M_fig5 = np.zeros((N_site_fig5+2,N_site_fig5+2))	
		for i in range(1,len(gate_H_M_fig5)-1):
			gate_H_M_fig5[i][i] = gate_epsilon - gate_Vg[0] - gate_sd*(i/(N_site_fig5+1))
			gate_H_M_fig5[i][i+1] = gate_H_M_fig5[i][i-1] = gate_V_M_arr[k] 
		gate_H_M_fig5[len(gate_H_M_fig5)-1][len(gate_H_M_fig5)-1] = -gate_sd 
		gate_H_M_fig5[0][1] = gate_H_M_fig5[1][0] = gate_V_A_arr[k][j]
		gate_H_M_fig5[len(gate_H_M_fig5)-1][len(gate_H_M_fig5)-2] = gate_H_M_fig5[len(gate_H_M_fig5)-2][len(gate_H_M_fig5)-1] = gate_V_A_arr[k][j] 
		AA_fig5, BB_fig5 = LA.eigh(np.real(gate_H_M_fig5))	
		eigval_store_fig5_case.append(AA_fig5.tolist())
		eigvec_store_fig5_case.append(BB_fig5.tolist())
	eigval_store_fig5.append(eigval_store_fig5_case)
	eigvec_store_fig5.append(eigvec_store_fig5_case)

"""
#######################################################


"""
Functions for multidim. index for DPB from individual 1-d indices and getting 1-d indices from multidim 
indices .. given only basis size for all dimensions 
"""


class INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH:
    """
    FINDING MULTIDIM. INDEX FOR DIRECT PRODUCT BASIS FROM 1-D INDICES ..
    FINDING INDIVIDUAL INDICES FROM MULTIDIM INDEX FOR THE DIRECT PRODUCT BASIS
    """

    def __init__(self, basis_size_arr):
        # array containing basis size for individual coordinates ##
        self.basis_size_arr = basis_size_arr
        # no of coordinates considered in eigenstate calculation ##
        self.N_dim = len(basis_size_arr)
    ##----------------------------------------------##

    def stride_arr(self):
        """
        preparing stride array
        """
        cur_index_pdt = 1
        for j in range(1, self.N_dim):
            cur_index_pdt *= self.basis_size_arr[j]
        # initializing stride array with first element ##
        stride_arr_init = [cur_index_pdt]
        ##------------------------------##
        """ other elements of stride array will be prepared from the first element of the stride array """
        cur_index_init = 1  # initializing current index for generating other elements of stride array ##
        while True:
            if cur_index_init == (self.N_dim-1):
                break
            else:
                cur_product = (
                    int(cur_index_pdt/self.basis_size_arr[cur_index_init]))
                cur_index_init += 1
                stride_arr_init.append(cur_product)
                cur_index_pdt = copy.deepcopy(cur_product)
        return stride_arr_init
    ##---------------------------------------------##

    def multidim_index_DPB(self, one_dim_index_arr):
        """ given one dimensional indices , returns multidimensional basis no """
        ### one_dim_index_arr has python indexing .. zero based indexing ###
        stride_arr = self.stride_arr()  # calling stride array ##
        multidim_basis_index = one_dim_index_arr[len(one_dim_index_arr)-1]+1
        for i in range(len(stride_arr)):
            multidim_basis_index += stride_arr[i]*one_dim_index_arr[i]
    ### --------- returning multidim basis index .. multidim index has one-based indexing ---------- ###
        return multidim_basis_index
    ##-------------------------------------------------------##

    def one_dim_indices(self, multidim_index):
        """ given multidim index for multidim DPB, returns individual one dimensional indices """
        ### ``` multidim_index ``` indexing starts from 1 ###
        stride_arr = self.stride_arr()  # generating object ##
        # subtracting 1 that is the last term in the sum to go from 1-d indices to final multidim index ##
        multidim_index_4_caln = multidim_index-1
        onedim_index_arr = []
        # multidim_index will change for finding each of the 1-d indices in the loop .. here it is first initialized ##
        multidim_index_cur = copy.deepcopy(multidim_index_4_caln)
        for i in range(len(stride_arr)):
            cur_onedim_index = multidim_index_cur//stride_arr[i]
            onedim_index_arr.append(cur_onedim_index)
            multidim_index_cur -= cur_onedim_index*stride_arr[i]
        onedim_index_arr.append(multidim_index_cur)
        ##-- Returns 1-dimensional index array .. zero based indexing -- ##
        return onedim_index_arr


#####################################################################################

class generate_R_tensor:
    def __init__(self, Tmat, Tmat_inv, BC, N, kappa, tau_c, temp, eigvalarr):
        self.Tmat = Tmat  # Transformation matrix from local basis to eigenbasis
        # Inverse of the transformation matrix from local basis to eigenbasis
        self.Tmat_inv = Tmat_inv
        self.BC = BC  # Boltamann constant
        self.N = N  # no of sites
        self.kappa = kappa
        self.tau_c = tau_c
        self.temp = temp
        self.eigvalarr = eigvalarr

    def bath_corrlnfunc(self, freq, direction):
        if direction == "plus":
            return (0.5*self.kappa)*(np.exp(-0.25*(freq**2)*(self.tau_c**2)))*np.exp(-(abs(freq)-freq)/(2*self.temp*self.BC))
        elif direction == "minus":
            return (0.5*self.kappa)*(np.exp(-0.25*(freq**2)*(self.tau_c**2)))*np.exp(-(abs(freq)+freq)/(2*self.temp*self.BC))

    def eval_Cmat(self, id1, id2, id3, id4):
        j = id1
        k = id2
        l = id3
        m = id4
        csum = 0.
        for mu in range(self.N+2):
            for nu in range(self.N+2):
                csum += self.Tmat[j][mu]*self.Tmat_inv[mu][l]*self.Tmat[k][nu] * \
                    self.Tmat_inv[nu][m] * \
                    (self.eigvalarr[mu]-self.eigvalarr[nu])
        return csum

    def eval_trm1(self, id1, id2, id3, id4):
        j = id1
        k = id2
        l = id3
        m = id4
        if k == m:
            sum1 = 0.+0j
            for n in range(1, self.N+1):
                if n == j:
                    for mu in range(self.N+2):
                        for nu in range(self.N+2):
                            freq1 = self.eigvalarr[nu]-self.eigvalarr[mu]
                            sum1 += self.Tmat[l][nu]*((self.Tmat[n][nu]))*(
                                (self.Tmat[n][mu])**2)*self.bath_corrlnfunc(freq1, "plus")
            return sum1
        else:
            return 0.+0j

    def eval_trm2(self, id1, id2, id3, id4):
        j = id1
        k = id2
        l = id3
        m = id4
        if j == l:
            sum2 = 0.+0j
            for n in range(1, self.N+1):
                if n == k:
                    for mu in range(self.N+2):
                        for nu in range(self.N+2):
                            freq2 = self.eigvalarr[nu]-self.eigvalarr[mu]
                            sum2 += self.Tmat[m][mu]*((self.Tmat[n][nu])**2)*(
                                self.Tmat[n][mu])*self.bath_corrlnfunc(freq2, "minus")
            return sum2
        else:
            return 0.+0j

    def eval_trm3(self, id1, id2, id3, id4):
        j = id1
        k = id2
        l = id3
        m = id4
        sum3 = 0.+0j
        for n in range(1, self.N+1):
            if n == m and n == k:
                for mu in range(self.N+2):
                    for nu in range(self.N+2):
                        freq3 = self.eigvalarr[nu] - self.eigvalarr[mu]
                        sum3 += self.Tmat[j][mu]*(self.Tmat[l][nu])*(
                            self.Tmat[n][mu])*self.Tmat[n][nu]*self.bath_corrlnfunc(freq3, "plus")
        return sum3

    def eval_trm4(self, id1, id2, id3, id4):
        j = id1
        k = id2
        l = id3
        m = id4
        sum4 = 0.+0j
        for n in range(1, self.N+1):
            if n == j and n == l:
                for mu in range(self.N+2):
                    for nu in range(self.N+2):
                        freq4 = self.eigvalarr[nu] - self.eigvalarr[mu]
                        sum4 += self.Tmat[m][mu]*(self.Tmat[k][nu])*(self.Tmat[n][mu])*(
                            self.Tmat[n][nu])*self.bath_corrlnfunc(freq4, "minus")
        return sum4


###################################################
###################################################
###################################################

"""
function for generating steady-state rate for a given 
set of parameters ....
When SD voltage and Gate voltage are tunred on ..
For different values of those parameters .. The 
Hamiltonian is different .. i.e. in the parameter space
AA(eigenvalues), BB(eigenvectors) and its inverse(BB_inv) 
will be different 
Other parameters that can vary are kappa(dephasing effect)
Temperature, relaxation time (tau_c_inv), No of sites
Redfield tensor object is generated for a given set of parameters 
(N_site, Temp, kappa, tau_c_inv, eigenvalues and eigenvector)

"""


def get_kss(strideobj, Rtensorobj, N_site):
    Cmat = np.zeros(((N_site+2)**2, (N_site+2)**2))
    R_tensor = np.zeros(((N_site+2)**2, (N_site+2)**2), dtype=complex)
    for i in range(len(R_tensor)):
        for j in range(len(R_tensor[i])):
            id1, id2 = strideobj.one_dim_indices(
                i+1)[0], strideobj.one_dim_indices(i+1)[1]
            id3, id4 = strideobj.one_dim_indices(
                j+1)[0], strideobj.one_dim_indices(j+1)[1]
            Cmat[i][j] = Rtensorobj.eval_Cmat(id1, id2, id3, id4)
            T1 = Rtensorobj.eval_trm1(id1, id2, id3, id4)
            T2 = Rtensorobj.eval_trm2(id1, id2, id3, id4)
            T3 = Rtensorobj.eval_trm3(id1, id2, id3, id4)
            T4 = Rtensorobj.eval_trm4(id1, id2, id3, id4)
            R_tensor[i][j] = -T1-T2+T3+T4

    Cmatnp1 = np.array(Cmat, dtype=complex)

    Cmatnp2 = -1j*Cmatnp1

    L = Cmatnp2 + R_tensor

    L1 = L/((h_eV/(2*np.pi)))  # Liouvillian

    J_vect = np.zeros([N_site+2, N_site+2])
    for elem in J_vect[:len(J_vect)-1]:
        elem[-1] = 0.5*gate_Gamma_a
    for j in range(0, len(J_vect)-1):
        J_vect[len(J_vect)-1][j] = 0.5*gate_Gamma_a

    J_vect[-1][-1] = gate_Gamma_a

    J_vect1 = J_vect.flatten()

    J_vect2 = np.diag(J_vect1)
    J_vect3 = J_vect2/(h_eV/(2*np.pi))

    L1 += -J_vect3

    B = np.zeros((N_site+2)**2)
    B[0] = -gate_J

    rho_sys = LA.solve(L1, B)
    rate_ss = gate_J/(rho_sys[0].real)
    return rate_ss


"""
dmarr = [N_site_fig5+2,N_site_fig5+2]
strideobj = INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH(dmarr)

for i in range(len(gate_V_A_arr)):
	for j in range(len(gate_V_A_arr[i])):
		eigvec_inv = LA.inv(eigvec_store_fig5[i][j])
		kss_arr = []
		for elem in gate_T_arr:	
			Rtensorobj = generate_R_tensor(eigvec_store_fig5[i][j],eigvec_inv,k_B_eV,N_site_fig5,gate_kappa,(1/gate_tau_c_inv),elem,eigval_store_fig5[i][j])
			rate_ss = get_kss(strideobj,Rtensorobj,N_site_fig5)
			kss_arr.append(math.log(rate_ss))
		slope = -linregress(gate_T_inv_fit,kss_arr)[0]
		print("{:d}	{:6.2f}		{:6.2f}		{:20.12e}".format(N_site_fig5,gate_V_M_arr[i],gate_V_A_M_ratio[j], slope*k_B_eV))



"""

"""
for i in range(len(N_bridge_fig6)):
	dmarr = [N_bridge_fig6[i]+2,N_bridge_fig6[i]+2]
	strideobj = INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH(dmarr)
	for j in range(len(gate_sd_arr)):
		eigvec_inv = LA.inv(eigvec_store_fig6[i][j])
		kss_arr = []
		kss_nobath_arr = []
		for elem in gate_T_arr:
			Rtensorobj = generate_R_tensor(eigvec_store_fig6[i][j],eigvec_inv,k_B_eV, N_bridge_fig6[i],gate_kappa,(1/gate_tau_c_inv),elem, eigval_store_fig6[i][j])
			Rtensorobj_nobath = generate_R_tensor(eigvec_store_fig6[i][j],eigvec_inv, k_B_eV, N_bridge_fig6[i], gate_kappa, 0., elem, eigval_store_fig6[i][j])
			rate_ss = get_kss(strideobj,Rtensorobj,N_bridge_fig6[i])
			rate_nobath_ss = get_kss(strideobj,Rtensorobj_nobath, N_bridge_fig6[i])
			kss_arr.append(math.log(rate_ss))
			kss_nobath_arr.append(math.log(rate_nobath_ss))
		slope = -linregress(gate_T_inv_fit, kss_arr)[0]
		slope_nobath = -linregress(gate_T_inv_fit, kss_nobath_arr)[0]
		print("{:d}	{:20.12e}	{:20.12e}	{:20.12e}".format(N_bridge_fig6[i], gate_sd_arr[j], slope*k_B_eV, slope_nobath*k_B_eV))

sys.exit()

"""


"""

for i in range(len(N_bridge_fig4)):
    dmarr = [N_bridge_fig4[i]+2, N_bridge_fig4[i]+2]
    strideobj = INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH(dmarr)
    for j in range(len(gate_Vg_arr)):
        eigvec_inv = LA.inv(eigvec_store_fig4[i][j])
        kss_arr = []
        kss_nobath_arr = []
        for elem in gate_T_arr:
            Rtensorobj = generate_R_tensor(
                eigvec_store_fig4[i][j], eigvec_inv, k_B_eV, N_bridge_fig4[i], gate_kappa, (1/gate_tau_c_inv), elem, eigval_store_fig4[i][j])
            Rtensorobj_nobath = generate_R_tensor(
                eigvec_store_fig4[i][j], eigvec_inv, k_B_eV, N_bridge_fig4[i], gate_kappa, 0., elem, eigval_store_fig4[i][j])
            rate_ss = get_kss(strideobj, Rtensorobj, N_bridge_fig4[i])
            rate_nobath_ss = get_kss(
                strideobj, Rtensorobj_nobath, N_bridge_fig4[i])
            kss_arr.append(math.log(rate_ss))
            kss_nobath_arr.append(math.log(rate_nobath_ss))
        slope = -linregress(gate_T_inv_fit, kss_arr)[0]
        slope_nobath = -linregress(gate_T_inv_fit, kss_nobath_arr)[0]
        print("{:d}	{:20.12e}	{:20.12e}	{:20.12e}".format(
            N_bridge_fig4[i], gate_Vg_arr[j], slope*k_B_eV, slope_nobath*k_B_eV))


sys.exit()
"""


for i in range(len(N_bridge)):
    dmarr = [N_bridge[i]+2, N_bridge[i]+2]
    strideobj = INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH(dmarr)
    eigvec_inv = LA.inv(eigvec_store[i][0])
    Rtensorobj = generate_R_tensor(
        eigvec_store[i][0], eigvec_inv, k_B_eV, N_bridge[i], gate_kappa_eV, (1/gate_tau_c_inv_eV), gate_T, eigval_store[i][0])
    Rtensorobj_nobath = generate_R_tensor(
        eigvec_store[i][0], eigvec_inv, k_B_eV, N_bridge[i], gate_kappa_eV, 0., gate_T, eigval_store[i][0])
    rate_ss = get_kss(strideobj, Rtensorobj, N_bridge[i])
    rate_ss_nobath = get_kss(strideobj, Rtensorobj_nobath, N_bridge[i])
    print("{:d}	{:6.2f}	{:6.2f}	{:20.12e}	{:20.12e}".format(
        N_bridge[i], gate_kappa, gate_tau_c_inv, rate_ss_nobath, rate_ss))

sys.exit()

"""

for i in range(len(N_bridge)):
    dmarr = [N_bridge[i]+2, N_bridge[i]+2]
    strideobj = INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH(dmarr)
    eigvec_inv = LA.inv(eigvec_store[i][0])
    kss_arr = []
    for elem in gate_T_arr:
        Rtensorobj = generate_R_tensor(
            eigvec_store[i][0], eigvec_inv, k_B_eV, N_bridge[i], gate_kappa, (1/gate_tau_c_inv), elem, eigval_store[i][0])
        rate_ss = get_kss(strideobj, Rtensorobj, N_bridge[i])
        kss_arr.append(math.log(rate_ss))
    #G_Q = 2.5*(1e-16)*rate_ss
    # conductance.append(G_Q)
    # print("{:d}	{:6.2f}	{:20.12e}".format(N_bridge[0],(1000/elem),G_Q))

    slope = -linregress(gate_T_inv_fit, kss_arr)[0]
    print("{:d}	{:20.12e}".format(N_bridge[i], slope*k_B_eV))
"""
sys.exit()

fig = plt.figure()
ax = plt.axes()
plt.plot(gate_T_inv, conductance, marker='o')
plt.ylim(1e-11, 1e-05)
plt.yscale('log')
ax.set_yticks([1e-11, 1e-09, 1e-07, 1e-05])
plt.show()
sys.exit()

#conductance_log = [math.log(conductance[i]) for i in range(len(conductance))]

# np.savetxt("./data_gate_fig3/GQ_N" +
#           str(N_bridge[0])+".dat", conductance, fmt='%20.12e')

sys.exit()


## activation energy calculation using linear regression ##

x = np.array(gate_T_inv).reshape(len(gate_T_inv), 1)
y = np.array(conductance_log)

model = LinearRegression()

model.fit(x, y)

coeff = model.coef_

print(coeff)
sys.exit()


###################################################
###################################################
###################################################

"""

## calculation of the Redfield tensor ##

# initialising the Redfield tensor as a matrix

Cmat = np.zeros(((N_bridge+2)**2, (N_bridge+2)**2))

R_tensor = np.zeros(((N_bridge+2)**2, (N_bridge+2)**2), dtype=complex)

dmarr = [N_bridge+2, N_bridge+2]

strideobj = INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH(dmarr)

BB_inv = LA.inv(BB)

Rtensorobj = generate_R_tensor(
    BB, BB_inv, k_B, N_bridge, gate_kappa_J, (1/gate_tau_c_inv_J), gate_T, AA)
#Rtensorobj = generate_R_tensor(BB,BB_inv, k_B, N_bridge, gate_kappa_J, 0., gate_T,AA)


for i in range(len(R_tensor)):
    for j in range(len(R_tensor[i])):
        id1, id2 = strideobj.one_dim_indices(
            i+1)[0], strideobj.one_dim_indices(i+1)[1]
        id3, id4 = strideobj.one_dim_indices(
            j+1)[0], strideobj.one_dim_indices(j+1)[1]
        Cmat[i][j] = Rtensorobj.eval_Cmat(id1, id2, id3, id4)
        T1 = Rtensorobj.eval_trm1(id1, id2, id3, id4)
        T2 = Rtensorobj.eval_trm2(id1, id2, id3, id4)
        T3 = Rtensorobj.eval_trm3(id1, id2, id3, id4)
        T4 = Rtensorobj.eval_trm4(id1, id2, id3, id4)
        R_tensor[i][j] = -T1-T2+T3+T4


Cmatnp1 = np.array(Cmat, dtype=complex)

Cmatnp2 = -1j*Cmatnp1

L = Cmatnp2 + R_tensor

L1 = L/((h_J/(2*np.pi)))  # Liouvillian

###################################################
### calculation specific for Segal-Nitzan paper ###
###################################################

## Steady-state Boundary conditions ##

J_vect = np.zeros([N_bridge+2, N_bridge+2])


for elem in J_vect[:len(J_vect)-1]:
    elem[-1] = 0.5*gate_Gamma_a

for j in range(0, len(J_vect)-1):
    J_vect[len(J_vect)-1][j] = 0.5*gate_Gamma_a

J_vect[-1][-1] = gate_Gamma_a

J_vect1 = J_vect.flatten()


J_vect2 = np.diag(J_vect1)
J_vect3 = J_vect2/(h_J/(2*np.pi))

L1 += -J_vect3

B = np.zeros((N_bridge+2)**2)
B[0] = -gate_J

rho_sys = LA.solve(L1, B)

#rate_ss = gate_Gamma_a*(rho_sys[-1].real/rho_sys[0].real)*(2*(np.pi)/h_J)

rate_ss = gate_J/(rho_sys[0].real)

print("{:d}   {:20.12e}".format(N_bridge, rate_ss))
#print("{:d}   {:6.2f}    {:20.12e}".format(N_bridge,(gate_kappa_J/gate_epsilon_J),rate_ss*(1e-04)))
#print("{:d}   {:6.2f}    {:20.12e}".format(N_bridge,(1e+03/(gate_T)),rate_ss))


"""

# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp
import traceback

METHOD_DEFAULT = 'RK23'
SEP_DEFAULT = True
ATOL_DEFAULT = 1e-6
RTOL_DEFAULT = 1e-3
tc = 0.1

xd = 0.295
X = 0.503
Ub = 1 # Voltage of infinite bus
U0_ori = 1.2096 # Voltage magnitude of generator bus
P_ori = 1 # Mechanical active power of generator

XI = xd+X
XII = 2.80
XIII = 1.041

delta0_ori = np.arcsin(1 / (1.41 / 0.798))
Ea_ori = 1 # E'
omega_0 = 1
omega_base = 2*np.pi*50
PMII_ori = 0.504
PMIII_ori = 1.35
D_default = 0. # Damping
TJ_default = 8.182
paras_default = (tc, PMII_ori, PMIII_ori, D_default, TJ_default)
tspan_default = np.arange(0,10,0.001)


def get_dP(t,y,tc,P,U0,D):
    E,delta_0,PMII,PMIII = return_PM_by_PV(P,U0,return_E_delta=True)
    if t<=tc:
        PM = PMII
    else:
        PM = PMIII
    dP = P-PM*np.sin(y[1])-D*(y[0]-1)
    return dP


def SMIB_2nd_atom(t,y,paras=paras_default):
    P, PM, D, TJ = paras
    w,delta = y
    ddelta = (w-1)*omega_base
    dw = 1/TJ*(P-PM*np.sin(delta)-D*(w-1))
    return [dw,ddelta]



def SMIB_2nd(t,y,paras=paras_default):
    tc, P, PMII, PMIII, D, TJ = paras
    if t<=tc:
        PM = PMII
    else:
        PM = PMIII
    dy1 = (y[0]-1)*omega_base
    dy0 = 1/TJ*(P-PM*np.sin(y[1])-D*(y[0]-1))
    return [dy0,dy1]


def judge_stable(delta,radian=True,maxDeg=500):
    if radian:
        degs = np.rad2deg(delta)
    else:
        degs = delta
    f = np.all([abs(degs)<maxDeg for x in delta])
    return f


def return_PM_by_PV(P,U0,return_E_delta=False):
    r = xd/X
    U0r = U0*(1+r)
    Ubr = Ub*r
    cos_phi_0_square = 1 - np.square(P * X / U0 / Ub)
    if cos_phi_0_square>=0:
        E = np.sqrt(np.square(U0r) + np.square(Ubr) - 2 * U0r * Ubr * np.sqrt(cos_phi_0_square))
        delta_0 = np.arcsin(P/(E*Ub/XI))
        PMII = E*Ub/XII
        PMIII = E*Ub/XIII
    else:
        E = delta_0 = PMII = PMIII = None
    if return_E_delta:
        return E,delta_0,PMII,PMIII
    else:
        return PMII, PMIII



def cal_delta_cr(P,U0,E=None,delta_0=None,PMII=None,PMIII=None):
    delta_h,cos_delta_cr = cal_cos_delta_cr(P,U0,E,delta_0,PMII,PMIII)
    delta_cr = np.arccos(cos_delta_cr)
    return delta_h,delta_cr

def cal_cos_delta_cr(P,U0,E=None,delta_0=None,PMII=None,PMIII=None):
    if any([x is None for x in [E,delta_0,PMII,PMIII]]):
        E,delta_0,PMII,PMIII = return_PM_by_PV(P,U0,return_E_delta=True)
    delta_h = np.pi - np.arcsin(P/PMIII)
    cos_delta_cr = (P*(delta_h-delta_0)+PMIII*np.cos(delta_h)-PMII*np.cos(delta_0))/(PMIII-PMII)
    return delta_h,cos_delta_cr


    
def simulate_SMIB_PV(tc, P, U0, D=D_default, TJ=TJ_default,sep=SEP_DEFAULT,
                     tspan=tspan_default,method=METHOD_DEFAULT):
    Ea, delta_0, PMII, PMIII = return_PM_by_PV(P, U0, return_E_delta=True)
    if sep:
        func_t = simulate_SMIB_tc_sep
    else:
        func_t = simulate_SMIB_tc
    sol = func_t(tc,P=P,PMII=PMII,PMIII=PMIII,
                 delta_0=delta_0,D=D,TJ=TJ, tspan=tspan,
                 method = method)
    return sol


def simulate_SMIB_tc(tc, P=P_ori,PMII=PMII_ori, PMIII=PMIII_ori, D=D_default, 
                     TJ=TJ_default, delta_0=delta0_ori,
                     tspan=tspan_default, method=METHOD_DEFAULT):
    paras_t = (tc, P, PMII, PMIII, D, TJ)
    y0 = [omega_0, delta_0]
    def SMIB_2nd_t(t, y):
        return SMIB_2nd(t, y, paras_t)

    sol = solve_ivp(SMIB_2nd_t, [min(tspan), max(tspan)], y0, t_eval=tspan, 
                    method=method,atol=ATOL_DEFAULT,rtol=RTOL_DEFAULT)
    return sol


my_sol = type('MY_SOL', (object,), dict(t=None,y=None))
def simulate_SMIB_tc_sep(tc, P=P_ori,PMII=PMII_ori, PMIII=PMIII_ori, 
                     D=D_default, TJ=TJ_default, delta_0=delta0_ori,
                     tspan=tspan_default,method=METHOD_DEFAULT):
    paras_II = (P,PMII,D,TJ)
    paras_III = (P,PMIII,D,TJ)
    def SMIB_2nd_II(t, y):
        return SMIB_2nd_atom(t, y, paras_II)
    def SMIB_2nd_III(t, y):
        return SMIB_2nd_atom(t, y, paras_III)
    
    y0_II = [omega_0, delta_0]
    tspan_II = tspan[tspan<tc]
    tspan_III = tspan[tspan>=tc]-tc

    if len(tspan_II)>1:

        sol_II = solve_ivp(SMIB_2nd_II, [min(tspan_II), max(tspan_II)], y0_II, t_eval=tspan_II, method=method,
                           atol=ATOL_DEFAULT,rtol=RTOL_DEFAULT)

        flag_success_II = sol_II.success
        y0_III = sol_II.y[:,-1]
    else:
        flag_success_II = True
        y0_III = y0_II
        sol_II = None

    if flag_success_II and len(tspan_III)>1:
        sol_III = solve_ivp(SMIB_2nd_III, [min(tspan_III), max(tspan_III)],
                            y0_III, t_eval=tspan_III, method=method,
                            atol=ATOL_DEFAULT,rtol=RTOL_DEFAULT)
        sol = sol_III
        if sol_II:
            sol.t = np.hstack([sol_II.t,sol_III.t+tc])
            sol.y = np.hstack([sol_II.y,sol_III.y])

    else:
        sol = sol_II

    if not sol:
        sol = my_sol()
        if len(tspan)<=0:
            ts = 0
        else:
            ts = min(tspan)
        sol.t = np.array(ts)
        sol.y = np.array([[omega_0],[delta_0]])
    
    return sol





#%%

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from scipy import interpolate
    tc = 0.212
    sol = simulate_SMIB_PV(tc, 1, 1.21,D=0)
    plt.clf()
    plt.plot(sol.t,sol.y.T)
    plt.show()
    delta_c = interpolate.griddata(sol.t,sol.y[1,:],tc)/np.pi*180

    print(delta_c)


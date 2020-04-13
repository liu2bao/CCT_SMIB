# -*- coding: utf-8 -*-
"""
Created on Sat May  4 15:32:27 2019

@author: LiuXianzhuang
"""
import sys
sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\PyPSASP')
import math
import matplotlib.pyplot as plt
import numpy as np
from SMIB_2nd import judge_stable, return_PM_by_PV, simulate_SMIB_PV, cal_cos_delta_cr, simulate_SMIB_tc_sep
from SMIB_2nd import D_default, TJ_default, U0_ori, P_ori, tspan_default, SEP_DEFAULT, METHOD_DEFAULT
from PyPSASP.PSASPClasses.PSASP import PSASP, func_change_t_regular, func_judge_stable_regular
from PyPSASP.constants import const as PSASPconst


FLAG_PLOT = False
FLAG_REC_OUTPUT = False
JUDGE_RANGE_DEFAULT = True
JUDGE_BY_ENERGY_FUNCTION = False

Tleft_default = 0
Tright_default = 0.2
Tstep_ib_default = 0.1
Tstep_ib_small = 0.002
Tsim_default = 5
eps_default = 1e-3
COUNT_DOUBLE_STEP = 5
T_RANGE_DEFAULT = (0, float('inf'))

EaKey = 'Ea'
U0Key = 'U0'
PKey = 'P'
Delta0Key = 'delta_0'
PMIIKey = 'PMII'
PMIIIKey = 'PMIII'
DKey = 'D'
TJKey = 'TJ'

OUTPUT_LF_KEY = 'output_lf'
OUTPUT_ST_LEFT_KEY = 'output_st_left'
OUTPUT_ST_RIGHT_KEY = 'output_st_right'
SUCCESS_LF_KEY = 'success_lf'
CCT_KEY = 'CCT'
TMAX_STEP_KEY = 'Tmax'
TSTEP_IB_LEFT_KEY = 'Tstep_ib_left'
TSTEP_IB_RIGHT_KEY = 'Tstep_ib_right'
JUDGE_RANGE_KEY = 'judge_range'

T_SIM_KEY = 'Tsim'
EPS_KEY = 'eps'
T_LEFT_KEY = 'tleft'
T_RIGHT_KEY = 'tright'
F_LEFT_KEY = 'fleft'
F_RIGHT_KEY = 'fright'
COUNT_ITER_KEY = 'count'
FLAG_LIMIT_TOUCHED_RIGHT_KEY = 'flag_limit_touched_right'
FLAG_LIMIT_TOUCHED_LEFT_KEY = 'flag_limit_touched_left'


def init_boundaries(func_simulate, func_judge_stable, t_left_ini=0, t_right_ini=Tright_default,
                    t_range=T_RANGE_DEFAULT, Tstep_ib_left=Tstep_ib_default, Tstep_ib_right=Tstep_ib_default,
                    max_try_ini=100, judge_range=JUDGE_RANGE_DEFAULT,
                    flag_rec_output=FLAG_REC_OUTPUT, label_var='tc'):
    labels = ['left', 'right']
    flag_labels = ['not stable', 'stable']
    if t_left_ini >= T_RANGE_DEFAULT[1] or t_left_ini <= T_RANGE_DEFAULT[0]:
        t_left_ini = Tleft_default
    if t_right_ini >= T_RANGE_DEFAULT[1] or t_right_ini <= T_RANGE_DEFAULT[0]:
        t_right_ini = Tright_default

    t_boundaries = [t_left_ini, t_right_ini]
    flag_limit_touched = [False, False]
    sols = [[], []]
    func_flag = [lambda s: s, lambda s: not s]
    judge_omni = lambda s1, s2: (s1 < t_range[0]) or (s1 <= 0) or (s1 > t_range[1]) or (s1 >= float('inf'))
    func_judge_outside = [lambda s1, s2: s1 <= s2, lambda s1, s2: s1 >= s2]
    # func_judge_outside = [judge_omni,judge_omni]
    step = [-Tstep_ib_left, Tstep_ib_right]

    for k in range(len(t_boundaries)):
        flag_out_range = False
        if not flag_limit_touched[k]:
            count_t = 0
            count_double_step = 0
            while count_t <= max_try_ini:
                if count_t == 0 and judge_range:
                    tt = t_range[k]
                else:
                    tt = t_boundaries[k]
                sol = func_simulate(tt)
                stable = func_judge_stable(sol)
                flag_t = func_flag[k](stable)
                if flag_t:
                    flag_limit_touched[k] = True
                    sols[k] = sol
                    if FLAG_PLOT:
                        plt.plot(sol.t, sol.y.transpose())
                        plt.title(tt)
                        plt.show()
                    if not (count_t == 0 and judge_range):
                        break
                else:
                    if count_t == 0 and judge_range:
                        flag_out_range = True
                        print('%s even when %s=%.5f' % (flag_labels[k], label_var, tt))
                        flag_limit_touched[1 - k] = True
                        t_boundaries[k] = t_range[k]
                        t_boundaries[1 - k] = t_range[k]
                        break
                    else:
                        t_boundaries[1 - k] = tt
                        flag_limit_touched[1 - k] = True
                        sols[1 - k] = sol
                print('%s %d(%d): %.5f' % (labels[k], count_t, flag_t, t_boundaries[k]))
                if not (count_t == 0 and judge_range):
                    t_boundaries[k] += step[k]
                    count_double_step += 1
                    if count_double_step >= COUNT_DOUBLE_STEP:
                        step[k] *= 2
                        count_double_step = 0
                count_t += 1
                if func_judge_outside[k](t_boundaries[k], t_range[k]):
                    break

        if flag_out_range:
            break

    rec_t = {T_LEFT_KEY: t_boundaries[0], T_RIGHT_KEY: t_boundaries[1],
             FLAG_LIMIT_TOUCHED_LEFT_KEY: flag_limit_touched[0],
             FLAG_LIMIT_TOUCHED_RIGHT_KEY: flag_limit_touched[1],
             TSTEP_IB_LEFT_KEY: Tstep_ib_left, TSTEP_IB_RIGHT_KEY: Tstep_ib_right}

    if flag_rec_output:
        rec_t.update({OUTPUT_ST_LEFT_KEY: sols[0], OUTPUT_ST_RIGHT_KEY: sols[1]})

    return rec_t


def calculate_CCT_dichotomy(func_simulate, func_judge_stable, rec_init,
                            label=None, eps=eps_default,
                            flag_rec_output=FLAG_REC_OUTPUT):
    if label is None:
        label = '-------AFFAIR-------'
    rec = rec_init.copy()
    rec[CCT_KEY] = rec[T_LEFT_KEY]
    rec[EPS_KEY] = eps
    rec[COUNT_ITER_KEY] = 0

    while abs(rec[T_LEFT_KEY] - rec[T_RIGHT_KEY]) > rec[EPS_KEY]:
        CT_t = (rec[T_LEFT_KEY] + rec[T_RIGHT_KEY]) / 2
        sol = func_simulate(CT_t)
        if FLAG_PLOT:
            plt.plot(sol.t.transpose(), sol.y.transpose())
            plt.title(str(CT_t))
            plt.show()
        stable = func_judge_stable(sol)
        if stable:
            rec[T_LEFT_KEY] = CT_t
            rec[F_LEFT_KEY] = stable
            rec[CCT_KEY] = CT_t
            if flag_rec_output:
                rec[OUTPUT_ST_LEFT_KEY] = sol
        else:
            rec[T_RIGHT_KEY] = CT_t
            rec[F_RIGHT_KEY] = stable
            if flag_rec_output:
                rec[OUTPUT_ST_RIGHT_KEY] = sol

        rec[COUNT_ITER_KEY] += 1
        print(
            '%s%d (%d,%.5f): %.5f, %.5f' % (label, rec[COUNT_ITER_KEY], stable, rec[T_RIGHT_KEY] - rec[T_LEFT_KEY],
                                            rec[T_LEFT_KEY], rec[T_RIGHT_KEY]))
    print('%sCCT = %.4f' % (label, rec[CCT_KEY]))

    return rec


def calculate_CCT_s(func_simulate, func_judge_stable, label=None, eps=eps_default,
                    t_left_ini=0, t_right_ini=Tright_default, t_range=(0, float('inf')),
                    Tstep_ib_left=Tstep_ib_default, Tstep_ib_right=Tstep_ib_default,
                    max_try_ini=100, judge_range=JUDGE_RANGE_DEFAULT,
                    flag_rec_output=FLAG_REC_OUTPUT, label_var='tc'):
    rec_t = init_boundaries(func_simulate, func_judge_stable, t_left_ini=t_left_ini, t_right_ini=t_right_ini,
                            t_range=t_range, Tstep_ib_left=Tstep_ib_left, Tstep_ib_right=Tstep_ib_right,
                            max_try_ini=max_try_ini, judge_range=judge_range,
                            flag_rec_output=flag_rec_output, label_var=label_var)

    if any([not rec_t[k] for k in [FLAG_LIMIT_TOUCHED_LEFT_KEY, FLAG_LIMIT_TOUCHED_RIGHT_KEY]]):
        rec_t[CCT_KEY] = float('nan')
        return rec_t
    elif abs(rec_t[T_LEFT_KEY] - rec_t[T_RIGHT_KEY]) <= eps:
        rec_t.update({CCT_KEY: rec_t[T_LEFT_KEY], EPS_KEY: eps})
        return rec_t
    rec_t = calculate_CCT_dichotomy(func_simulate, func_judge_stable,
                                    rec_init=rec_t, label=label, eps=eps,
                                    flag_rec_output=flag_rec_output)
    return rec_t


def judge_stable_SMIB(sol):
    f = judge_stable(sol.y[1, :])
    return f



def calculate_CCT_SMIB_by_EnergyFunction(P, U0, TJ=TJ_default, Ea=None, delta_0=None, PMII=None, PMIII=None):
    delta_h, cos_delta_cr = cal_cos_delta_cr(P, U0, Ea, delta_0, PMII, PMIII)
    if np.cos(delta_0) <= cos_delta_cr:
        CCT = float('nan')
        t_left = t_right = 0
        flag_limit_touched_left = False
        flag_limit_touched_right = True
        print('not stable even if tc=0')
    else:
        flag_limit_touched_left = True
        solt_inf = simulate_SMIB_tc_sep(float('inf'),P=P,PMII=PMII,PMIII=PMIII,D=0,TJ=TJ,delta_0=delta_0)
        idx_out, = np.where(np.cos(solt_inf.y[1, :]) <= cos_delta_cr)
        if len(idx_out) > 0:
            idx_out_min = np.min(idx_out)
            if idx_out_min >= 1:
                idx_out_min_left = idx_out_min-1
                t_left = solt_inf.t[idx_out_min_left]
                CCT = t_left
                t_right = solt_inf.t[idx_out_min]
                flag_limit_touched_right = True
            else:
                CCT = float('nan')
                t_left = t_right = 0
                flag_limit_touched_left = False
                flag_limit_touched_right = True
                print('not stable even if tc=0')
        else:
            CCT = float('inf')
            t_left = t_right = CCT
            flag_limit_touched_right = False
            print('stable even if tc=inf')
    rec = {CCT_KEY: CCT, T_LEFT_KEY:t_left, T_RIGHT_KEY:t_right,
           FLAG_LIMIT_TOUCHED_LEFT_KEY: flag_limit_touched_left,
           FLAG_LIMIT_TOUCHED_RIGHT_KEY: flag_limit_touched_right}
    return rec



def calculate_CCT_SMIB_2(U0=U0_ori, P=P_ori, D=D_default, TJ=TJ_default, label=None,
                         eps=eps_default, t_left_ini=0, t_right_ini=Tright_default, t_range=(0, float('inf')),
                         Tstep_ib_left=Tstep_ib_default, Tstep_ib_right=Tstep_ib_default,
                         max_try_ini=100, judge_range=JUDGE_RANGE_DEFAULT, tspan=tspan_default,
                         flag_rec_output=FLAG_REC_OUTPUT, judge_by_energyfunction=JUDGE_BY_ENERGY_FUNCTION,
                         sep=SEP_DEFAULT, method=METHOD_DEFAULT):
    Ea, delta_0, PMII, PMIII = return_PM_by_PV(P, U0, return_E_delta=True)
    paras = {U0Key: U0, PKey: P, Delta0Key: delta_0, EaKey: Ea, PMIIKey: PMII, PMIIIKey: PMIII, DKey: D, TJKey: TJ}
    if any([(x is None) or (math.isnan(x)) for x in paras.values()]):
        rec = {SUCCESS_LF_KEY: False}
        CCT = None
        print('Load flow does not exist.')
    else:
        Jt = D==0 and judge_by_energyfunction
        def func_sim_t(tc):
            solt = simulate_SMIB_PV(tc, P, U0, D=D, TJ=TJ, sep=sep, tspan=tspan, method=method)
            return solt
        if Jt:
            rec = calculate_CCT_SMIB_by_EnergyFunction(P, U0, TJ=TJ, Ea=Ea, delta_0=delta_0, PMII=PMII, PMIII=PMIII)
            print('CCT = '+str(rec[CCT_KEY]))
        else:

            rec = calculate_CCT_s(func_sim_t, judge_stable_SMIB, label=label, eps=eps,
                                  t_left_ini=t_left_ini, t_right_ini=t_right_ini,
                                  t_range=t_range, Tstep_ib_left=Tstep_ib_left, Tstep_ib_right=Tstep_ib_right,
                                  max_try_ini=max_try_ini, judge_range=judge_range,
                                  flag_rec_output=flag_rec_output)
        CCT = rec[CCT_KEY]
        rec[SUCCESS_LF_KEY] = True
    return paras, rec, CCT


def find_P_boundaries(U0=U0_ori, D=D_default, TJ=TJ_default, label=None, tspan=tspan_default,
                      eps=eps_default, sep=SEP_DEFAULT, method=METHOD_DEFAULT):
    def func_sim_t(Pt, tc):
        Ea, delta_0, PMII, PMIII = return_PM_by_PV(Pt, U0, return_E_delta=True)
        paras = {U0Key: U0, PKey: Pt, Delta0Key: delta_0, EaKey: Ea, PMIIKey: PMII, PMIIIKey: PMIII, DKey: D, TJKey: TJ}
        if any([(x is None) or (math.isnan(x)) for x in paras.values()]):
            return None
        solt = simulate_SMIB_PV(tc, Pt, U0, D=D, TJ=TJ, sep=sep, tspan=tspan, method=method)
        return solt

    def judge_stable_t(sol_t):
        stable_t = False
        if sol_t:
            stable_t = judge_stable_SMIB(sol_t)
        return stable_t

    rec_left = calculate_CCT_s(lambda Ptt: func_sim_t(Ptt, float('inf')), judge_stable_t, label=label, eps=eps,
                               label_var='P')
    P_left = rec_left[T_RIGHT_KEY]
    flag_limit_touched_left = rec_left[FLAG_LIMIT_TOUCHED_RIGHT_KEY]
    rec_right = calculate_CCT_s(lambda Ptt: func_sim_t(Ptt, 0), judge_stable_t, label=label, eps=eps, label_var='P')
    P_right = rec_right[T_LEFT_KEY]
    flag_limit_touched_right = rec_right[FLAG_LIMIT_TOUCHED_LEFT_KEY]

    return P_left, P_right, flag_limit_touched_left, flag_limit_touched_right


def calculate_CCT_SMIB_PSASP(path_temp, U0=U0_ori, P=P_ori, label=None,
                             eps=eps_default, t_left_ini=0, t_right_ini=Tright_default, t_range=(0, 9999),
                             Tstep_ib_left=Tstep_ib_default, Tstep_ib_right=Tstep_ib_default,
                             max_try_ini=100, judge_range=JUDGE_RANGE_DEFAULT,
                             flag_rec_output=FLAG_REC_OUTPUT):
    PSASP_t = PSASP(path_temp)
    gen_s = PSASP_t.parser.parse_single_s(PSASPconst.LABEL_LF, PSASPconst.LABEL_SETTINGS, PSASPconst.LABEL_GENERATOR)
    for hh in range(len(gen_s)):
        type_t = gen_s[hh][PSASPconst.CtrlModeKey]
        if type_t == -1:
            # gen_s[hh][PSASPconst.GenPgKey] = gen_s[hh][PSASPconst.PmaxKey]*P
            gen_s[hh][PSASPconst.GenPgKey] = P
            gen_s[hh][PSASPconst.V0Key] = U0
            break
    PSASP_t.writer.write_to_file_s_lfs_autofit(gen_s)
    rec = {}
    rec[SUCCESS_LF_KEY] = PSASP_t.calculate_LF()
    if not rec[SUCCESS_LF_KEY]:
        CCT = None
        print('Load flow does not exist.')
    else:
        def func_sim_t(tc):
            func_change_t_regular(PSASP_t, tc)
            solt = None
            s_st = PSASP_t.calculate_ST()
            if s_st:
                solt = PSASP_t.parser.get_gen_angles_st()
            return solt

        def func_judge_stable_t(sol):
            if sol:
                return func_judge_stable_regular(PSASP_t)
            else:
                return False

        rec = calculate_CCT_s(func_sim_t, func_judge_stable=func_judge_stable_t, label=label, eps=eps,
                              t_left_ini=t_left_ini, t_right_ini=t_right_ini,
                              t_range=t_range, Tstep_ib_left=Tstep_ib_left, Tstep_ib_right=Tstep_ib_right,
                              max_try_ini=max_try_ini, judge_range=judge_range,
                              flag_rec_output=flag_rec_output)
        CCT = rec[CCT_KEY]
    paras = {PKey: P, U0Key: U0}
    return paras, rec, CCT


# %%
if __name__ == '__main__':
    path_temp_t = r'E:\01_Research\98_Data\SmallSystem_PSASP\SMIB\SMIB_0'
    paras, rec, CCT = calculate_CCT_SMIB_PSASP(path_temp_t, P=4.5)
    P_left, P_right, flag_limit_touched_left, flag_limit_touched_right = find_P_boundaries(1.21)
    pass
    # rec_t = calculate_CCT_SMIB_2(U0=0.82,P=1.36,D=0)
    # calculate_CCT_SMIB_2(0.80670, 1.42857)
    # calculate_CCT_SMIB_2(0.81172, 1.42857)
    # calculate_CCT_SMIB_2(0.81674, 1.42857)

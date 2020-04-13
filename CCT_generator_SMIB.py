from Calculate_CCT import calculate_CCT_SMIB_2, find_P_boundaries
from Calculate_CCT import CCT_KEY,EaKey,Delta0Key,U0Key,PKey,T_LEFT_KEY,T_RIGHT_KEY,DKey
from Calculate_CCT import SUCCESS_LF_KEY,FLAG_LIMIT_TOUCHED_RIGHT_KEY,FLAG_LIMIT_TOUCHED_LEFT_KEY,Tstep_ib_small
from Calculate_CCT import JUDGE_RANGE_KEY,TSTEP_IB_LEFT_KEY,TSTEP_IB_RIGHT_KEY,JUDGE_BY_ENERGY_FUNCTION
from SMIB_2nd import D_default
import numpy as np
import utils_sqlite
from math import isnan
import os
import traceback

patht,filet = os.path.split(os.path.abspath(__file__))

dbs_folder = os.path.join(patht,'dbs')
db_cct = os.path.join(dbs_folder,'CCT_SMIB.db')
table_records = 'records'

ParasKey = 'paras'
RecKey = 'rec'

tol_dist = 0.0001
Nsave = 10

NP = 400
range_P = [0,2]
NU0 = 200
range_U0 = [0.5,1.5]

# NU0 = 0
# range_U0 = [1.21,1.21]

ls_U0_default = np.linspace(*range_U0, NU0 + 1)
ls_P_default = np.linspace(*range_P, NP+1)
step_P_default = 0.001

def initial_condition_generator(ls_U0=ls_U0_default, ls_P=ls_P_default, Dn=D_default):
    for U0 in ls_U0:
        for P in ls_P:
            yield {U0Key:U0,PKey:P,DKey:Dn}


def initial_condition_generator_valid(ls_U0=ls_U0_default,step_P=step_P_default,Dn=D_default):
    for U0 in ls_U0:
        P_left,P_right,flag_limit_touched_left,flag_limit_touched_right = find_P_boundaries(U0,Dn)
        print('-----------------------------------P=[%.5f,%.5f]-----------------------------------' % (P_left,P_right))
        ls_P = np.arange(P_left,P_right,step_P)
        for P in ls_P:
            yield {U0Key:U0,PKey:P,DKey:Dn}





def cal_distance_0(para_1,para_2):
    s = max([abs(para_1[k]-para_2[k]) for k in set(para_1.keys()).intersection(set(para_2.keys()))])
    return s

def cal_distance(para_1,para_2):
    s = np.sqrt(sum([np.square(para_1[k]-para_2[k]) for k in para_1.keys() if k in para_2.keys()]))
    return s


def judge_calculated(para_t, paras_old, func_cal_dist, tol_dist, count_start=0, return_count=False):
    flag_done = False
    count = count_start
    min_dist = float('inf')
    idx_nearest = None
    if paras_old:
        for hh in range(count_start,len(paras_old)):
            para_old = paras_old[hh]
            dist_t = func_cal_dist(para_t,para_old)
            if dist_t<min_dist:
                idx_nearest = hh
                min_dist = dist_t
            if dist_t<=tol_dist:
                flag_done = True
                count = hh
                break
        if not flag_done:
            for hh in range(0,count_start):
                para_old = paras_old[hh]
                dist_t = func_cal_dist(para_t,para_old)
                if dist_t<min_dist:
                    idx_nearest = hh
                    min_dist = dist_t
                if dist_t<=tol_dist:
                    flag_done = True
                    count = hh
                    break
    if return_count:
        return flag_done,count,idx_nearest,min_dist
    else:
        return flag_done,idx_nearest,min_dist


def judge_calculated_s(para_t, paras_old, tol_dist, keys_cal):
    flag_done = False
    if paras_old:
        A = np.ones([len(paras_old),1])*np.mat([para_t[k] for k in keys_cal])
        B = np.mat([[t[k] for k in keys_cal] for t in paras_old])
        dists = np.sqrt(np.sum(np.square(A-B),1))
        minDist = min(dists)
        flag_done = minDist<=tol_dist

    return flag_done

def generate_CCT_SMIB(db=db_cct, table=table_records, para_gen=initial_condition_generator,read_records=True,
                      start_from_last_calDist=True, ini_from_nearest_calCCT=True,start_count=0,
                      judge_by_energyfunction=JUDGE_BY_ENERGY_FUNCTION ):
    paras_old = []
    records = []
    if read_records:
        records = utils_sqlite.read_db(db,table,return_dict_form=True)
        if records:
            paras_old = [x[ParasKey] for x in records]

    keys_check_ini_valid = {SUCCESS_LF_KEY, FLAG_LIMIT_TOUCHED_LEFT_KEY, FLAG_LIMIT_TOUCHED_RIGHT_KEY,CCT_KEY}
    records_save = []
    paras_cand = para_gen()
    count = 0
    count_start = 0
    for para_t in paras_cand:
        if count<start_count:
            count+=1
            continue
        if start_from_last_calDist:
            flag_done,count_start,idx_nearest,min_dist = \
                judge_calculated(para_t,paras_old,cal_distance_0,tol_dist,count_start,return_count=True)
        else:
            flag_done,idx_nearest,min_dist = judge_calculated(para_t,paras_old,cal_distance_0,tol_dist)
        # flag_done = judge_calculated_s(para_t,paras_old,tol_dist,keys_cal=(PKey,U0Key))
        flag_do = not flag_done
        if flag_do:
            if ini_from_nearest_calCCT and idx_nearest:
                record_old_t = records[idx_nearest]
                if RecKey in record_old_t.keys():
                    rec_old_t = record_old_t[RecKey]
                    keys_rec_old_t =  rec_old_t.keys()
                    if not set(keys_check_ini_valid).difference(keys_rec_old_t):
                        CCT_old_t = rec_old_t[CCT_KEY]
                        if (CCT_old_t is not None) and (not isnan(CCT_old_t)):
                            if all([rec_old_t[k] for k in keys_check_ini_valid]):
                                if T_LEFT_KEY in keys_rec_old_t:
                                    para_t['t_left_ini'] = rec_old_t[T_LEFT_KEY]
                                if T_RIGHT_KEY in keys_rec_old_t:
                                    para_t['t_right_ini'] = rec_old_t[T_RIGHT_KEY]
                                para_t[TSTEP_IB_LEFT_KEY] = Tstep_ib_small
                                para_t[TSTEP_IB_RIGHT_KEY] = Tstep_ib_small
                                # para_t[JUDGE_RANGE_KEY] = False
            try:
                para_t['judge_by_energyfunction'] = judge_by_energyfunction
                paras_t,rec_t,CCT = calculate_CCT_SMIB_2(**para_t)
                record_t = {utils_sqlite.KeyToken:utils_sqlite.gen_token(),
                            ParasKey:paras_t,RecKey:rec_t,CCT_KEY:CCT}
                record_t.update(para_t)
                paras_old.append(paras_t)
                records.append(record_t)
                records_save.append(record_t)
                print(str(count) + ' : '+str(para_t)+' calculated.')
            except:
                traceback.print_exc()
                print(str(count) + ' : '+str(para_t)+' failed.')
        else:
            print(str(count)+' : Record with similar parameters already exists!')

        if len(records_save)>=Nsave:
            utils_sqlite.insert_from_list_to_db(db,table,list_keys=None,list_data=records_save)
            records_save = []
        count+=1

    utils_sqlite.insert_from_list_to_db(db, table, list_keys=None, list_data=records_save,
                                        primary_key=utils_sqlite.KeyToken)


if __name__=='__main__':
    from Calculate_CCT import T_LEFT_KEY,EPS_KEY
    db = db_cct
    # db = r'CCT_SMIB(D=0).db'
    table = table_records
    records = utils_sqlite.read_db(db, table, return_dict_form=True)
    data = [(x[utils_sqlite.KeyToken], x[CCT_KEY], x[RecKey]) for x in records]
    data_alter = []
    for d in records:
        rec_t = d[RecKey]
        if FLAG_LIMIT_TOUCHED_LEFT_KEY in rec_t.keys() and FLAG_LIMIT_TOUCHED_RIGHT_KEY in rec_t.keys():
            if rec_t[FLAG_LIMIT_TOUCHED_LEFT_KEY] and rec_t[FLAG_LIMIT_TOUCHED_RIGHT_KEY]:
                if d[CCT_KEY] is None or isnan(d[CCT_KEY]):
                    if abs(rec_t[T_RIGHT_KEY]-rec_t[T_LEFT_KEY])<=rec_t[EPS_KEY]:
                        CCT_real = rec_t[T_LEFT_KEY]
                        rec_t[CCT_KEY] = CCT_real
                        d[RecKey] = rec_t
                        d[CCT_KEY] = CCT_real
                        data_alter.append(d)

    tokens = [x[utils_sqlite.KeyToken] for x in data_alter]
    keys_t,data_t = utils_sqlite.formulate_list_of_dicts(data_alter)
    utils_sqlite.update_list_to_db_multiref(db,table,keys_t,data_t,utils_sqlite.KeyToken,tokens)

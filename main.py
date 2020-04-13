from CCT_generator_SMIB import generate_CCT_SMIB,initial_condition_generator,initial_condition_generator_valid
from CCT_generator_SMIB import dbs_folder,step_P_default
from SMIB_2nd import METHOD_DEFAULT,SEP_DEFAULT
from Calculate_CCT import DKey
import sys,os
import numpy as np

if __name__=='__main__':
    # generate_CCT_SMIB(os.path.join('dbs','temp.db'), read_records=False, judge_by_energyfunction=True)
    paras = {'D':0.,'pl':0.,'pr':2.,'ps':0.001,'u0l':0.5,'u0r':1.5,'u0s':0.001,'rr':0,'ef':0,'prefix':''}
    keys_paras = list(paras.keys())
    cf_paras = {k:type(v) for k,v in paras.items()}
    count = 0
    for k in range(1,len(sys.argv)):
        key_t = keys_paras[k-1]
        paras[key_t] = cf_paras[key_t](sys.argv[k])

    D,pl,pr,ps,u0l,u0r,u0s,rr,ef,prefix = list(paras.values())

    para_gen = lambda :initial_condition_generator(Dn=D,ls_U0=np.arange(u0l,u0r,u0s),ls_P=np.arange(pl,pr,ps))

    db_name = 'CCT_SMIB(D=%.2f)_%s_%s.db' % (D,str(SEP_DEFAULT),METHOD_DEFAULT)
    if ef:
        db_name = 'EF_'+db_name
    if prefix:
        db_name = prefix+'_'+db_name
    db = os.path.join(dbs_folder,db_name)
    generate_CCT_SMIB(para_gen=para_gen,db=db,read_records=rr,judge_by_energyfunction=ef)


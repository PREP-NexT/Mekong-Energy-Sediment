# @Time : 2025/4/1 11:58
# @Author : XU BO


# from pyomo.environ import SolverFactory
# from pyomo.opt import SolverStatus, TerminationCondition
import configparser
import time
import pandas as pd
from prepshot import load_data
from prepshot.model_gurobi_2 import creat_model
from prepshot import utils

# default paths
inputpath = '../scenario_inputs/'
outputpath = './output/Uncertainty/'
logpath = './log/'

# load global parameters
config = configparser.RawConfigParser(inline_comment_prefixes="#")
config.read('global_uncertainty.properties')
basic_para = dict(config.items('global parameters'))
solver_para = dict(config.items('solver parameters'))
hydro_para = dict(config.items('hydro parameters'))

# global parameters
time_length = int(basic_para['hour'])
month = int(basic_para['month'])
start_year = int(basic_para['start_year'])
dt = int(basic_para['dt'])
start = int(basic_para['start'])
elec_price = float(basic_para['elec_price'])
# Fraction of One Year of Modeled Timesteps
weight = (month * time_length * dt) / 8760

# hydro parameters
ishydro = int(hydro_para['ishydro'])
sed_con = 10
error_threshold = float(hydro_para['error_threshold'])
iteration_number = int(hydro_para['iteration_number'])

import sys
start_num = sys.argv[1]
end_num = sys.argv[2]
selected_files = list(range(int(start_num),int(end_num)+1))

for i in selected_files:
    input_filename = inputpath + f"input_{i}.xlsx"
    solutions = pd.read_excel(input_filename,sheet_name='portfolio',index_col=0)
    hydro_potfolio = [int(a) for a in list(solutions.iloc[0])[:-2]]
    trans_limit = int(pd.read_excel(input_filename,sheet_name='trans_limit',header=None).iloc[0, 1])
    import_limit = pd.read_excel(input_filename,sheet_name='import_limit',header=None).iloc[0, 1]
    # solver config
    start_time = time.time()
    logfile = logpath + "main_%s"%time.strftime("%Y-%m-%d-%H-%M-%S")+'_%s'%i+'.log'
    def write(message, file=logfile):
        with open(file, "a") as f:
            f.write(message)
            f.write("\n")
    # load data

    output_filename = outputpath + basic_para['outputfile']+'_%s'%i
    start_time = time.time()
    para  = load_data(input_filename, month, time_length)

    write("Starting load parameters ...")
    write("Set parameter solver to value %s"%basic_para['solver'])
    write("Set parameter timelimit to value %s"%int(solver_para['timelimit']))
    write("Set parameter input_filename to value %s"%input_filename)
    write("Set parameter output_filename to value %s.nc"%output_filename)
    write("Set parameter time_length to value %s"%basic_para['hour'])
    write("Parameter loading completed, taking %s minutes"%(round((time.time() - start_time)/60,1)))
    write("\n=========================================================\n")
    write("Starting load data ...")
    write("Data loading completed, taking %s minutes"%(round((time.time() - start_time)/60,1)))
    write("\n=========================================================\n")


    # creating model
    write("Start creating model ...")
    start_time = time.time()
    # specify the dam portfolio/ start solution

    model,result = creat_model(para, weight, dt= dt , ishydro= ishydro, Sed_Cons= sed_con, start=start,
                               hydro_portfolio=hydro_potfolio,trans_limit=trans_limit,elec_price=elec_price,import_limit=import_limit)
    write("creatings model completed, taking %s minute"%(round((time.time() - start_time)/60,1)))
    write("\n=========================================================\n")
    # model parameters
    model.Params.LogToConsole = True # 显示求解过程
    model.Params.LogFile = logpath + "gurobi_%s.log"%time.strftime("%Y-%m-%d-%H-%M-%S")
    model.Params.MIPGap = float(solver_para['gap']) # 百分比界差
    model.Params.TimeLimit = int(solver_para['timelimit']) # 限制求解时间为 100s


    # model optimization
    write("Start solving model  ...")
    start_time = time.time()
    model.optimize()
    write("Solving model completed, taking %s minutes"%(round((time.time() - start_time)/60,1)))
    write("\n=========================================================\n")


    write('finish   ')
    print('finish-----------',i)


    # write output
    utils.saveresult_gurobi(result, output_filename, ishydro=ishydro)



print('finish  all ')
# @Time : 2022/6/8 11:58
# @Author : XU BO


import configparser
import time
import pandas as pd
from prepshot import load_data
from prepshot.model_gurobi_2 import creat_model
from prepshot import utils
import argparse

# carbon emission, sediment, transmission_line scenarios
parser = argparse.ArgumentParser(description='scenario approach')
parser.add_argument('--carbon', type=int, help='carbon emission limits scenario')
parser.add_argument('--sediment', type=float, default=0, help='sediment constrainst scenarios, 0--all, 100--none')
parser.add_argument('--limit', type=int, default=1, help='transmission line scenarios')
parser.add_argument('--ishydro', type=int, default=1, help='transmission line scenarios')
parser.add_argument('--hydro_portfolio', type=int, default=0, help='transmission line scenarios')
args = parser.parse_args()


# default paths
inputpath = './input/'
outputpath = './output/'
logpath = './log/'
logfile = logpath + "main_%s"%time.strftime("%Y-%m-%d-%H-%M-%S")+'_carbon%s'%args.carbon+'_limit%s'%args.limit+'_sediment%s'%args.sediment+'.log'

def write(message, file=logfile):
    with open(file, "a") as f:
        f.write(message)
        f.write("\n")

write("Starting load parameters ...")
start_time = time.time()

# load global parameters
config = configparser.RawConfigParser(inline_comment_prefixes="#")
config.read('global.properties')
basic_para = dict(config.items('global parameters'))
solver_para = dict(config.items('solver parameters'))
hydro_para = dict(config.items('hydro parameters'))

# global parameters
time_length = int(basic_para['hour'])
month = int(basic_para['month'])
start_year = int(basic_para['start_year'])
dt = int(basic_para['dt'])
start = int(basic_para['start'])
weight = (month * time_length * dt) / 8760

# hydro parameters
error_threshold = float(hydro_para['error_threshold'])
iteration_number = int(hydro_para['iteration_number'])

# scenarios
trans_limit= args.limit
ishydro = args.ishydro
sed_con = args.sediment
input_filename = inputpath + basic_para['inputfile']+'%s'%args.carbon+'.xlsx'
output_filename = outputpath + basic_para['outputfile']+'%s'%args.carbon+'_limit%s'%args.limit+'_sediment%s'%args.sediment

# solver config
# solver = SolverFactory(basic_para['solver'], solver_io='python')
# solver.options['TimeLimit'] = int(solver_para['timelimit'])
# solver.options['LogToConsole'] = 0
# solver.options['LogFile'] = logpath + "gurobi_%s.log"%time.strftime("%Y-%m-%d-%H-%M-%S")


write("Set parameter solver to value %s"%basic_para['solver'])
write("Set parameter timelimit to value %s"%int(solver_para['timelimit']))
write("Set parameter input_filename to value %s"%input_filename)
write("Set parameter output_filename to value %s.nc"%output_filename)
write("Set parameter time_length to value %s"%basic_para['hour'])
write("Parameter loading completed, taking %s minutes"%(round((time.time() - start_time)/60,1)))
write("\n=========================================================\n")
write("Starting load data ...")
start_time = time.time()

# load data
para  = load_data(input_filename, month, time_length)
write("Data loading completed, taking %s minutes"%(round((time.time() - start_time)/60,1)))
write("\n=========================================================\n")



# Create model
write("Start creating model ...")
start_time = time.time()
model,result = creat_model(para, weight, dt= dt , ishydro= ishydro, Sed_Cons= sed_con, start=start, trans_limit=trans_limit)
write("creatings model 1 completed, taking %s minute"%(round((time.time() - start_time)/60,1)))
write("\n=========================================================\n")

# model parameters
model.Params.LogToConsole = True
model.Params.LogFile = logpath + "gurobi_%s"%time.strftime("%Y-%m-%d-%H-%M-%S")+'_carbon%s'%args.carbon+'_limit%s'%args.limit+'_sediment%s'%args.sediment+'.log'
model.Params.MIPGap = float(solver_para['gap'])
model.Params.TimeLimit = int(solver_para['timelimit'])


# model optimization
write("Start solving model  ...")
start_time = time.time()
model.optimize()
model.computeIIS()
model.write("model.ilp")
write("Solving model completed, taking %s minutes"%(round((time.time() - start_time)/60,1)))
write("\n=========================================================\n")


write('finish solving the model ~~~~~~~~~~~ ')
print('finish solving the model ~~~~~~~~~~~ ')



# write output
utils.saveresult_gurobi(result, output_filename, ishydro=ishydro)



write("Finish!")
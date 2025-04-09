# @Time : 2022/11/2 9:26
# @Author : XU BO
# @Time : 2022/6/5 19:36
# @Author : XU BO
import numpy as np
import gurobipy as gp
import pandas as pd
def creat_model(para, weight ,dt=1, ishydro=1, Sed_Cons=0, start=0, start_variables={}, hydro_portfolio=[],trans_limit=0, elec_price=87):
    # ishydro  0:fixed hydro input(LP),   1:optimiz hydro construct and operation(MILP),  2:only optimize opration, dam portfolio must be given(LP)
    # Sed_Cons    0: no constraints    >0 have constraints
    # start   0:no start solution    1: start solution containing all variables    2: start solution containing only binary variables
    # if start!=0  start_variables must be given
    model= gp.Model()
    result={}
    # sets and parameters
    year = para['year_sets']
    zone = para['zone_sets']
    zone_ASEAN = ['Cambodia', 'Laos', 'Myanmar', 'Thailand', 'Vietnam']
    tech = para['tech_sets']
    hour = para['hour_sets']
    hour_p = [0]+para['hour_sets']
    month = para['month_sets']

    # add index by type
    if 'storage' in para['type'].values():
        storage_tech = [i for i, j in para['type'].items() if j == 'storage']
    else:
        storage_tech = 0

    if 'nondispatchable' in para['type'].values():
        nondispatchable_tech = [i for i, j in para['type'].items() if j == 'nondispatchable']
    else:
        nondispatchable_tech = 0

    if 'dispatchable' in para['type'].values():
        dispatchable_tech = [i for i, j in para['type'].items() if j == 'dispatchable']
    else:
        dispatchable_tech = 0
    year_zone_zone_tuples = [(y, z, z1) for y in year for z in zone for z1 in zone if
                             (z != z1 and not np.isnan(para['transmission'][z, z1]))]
    year_zone_zone_ASEAN_tuples = [(y, z, z1) for y in year for z in zone_ASEAN for z1 in zone_ASEAN if
                             (z != z1 and not np.isnan(para['transmission'][z, z1]))]
    hour_month_year_zone_zone_tuples = [(h, m, y, z, z1) for h in hour for m in month for y, z, z1 in
                                        year_zone_zone_tuples]
    hour_month_year_zone_zone_ASEAN_tuples = [(h, m, y, z, z1) for h in hour for m in month for y, z, z1 in
                                        year_zone_zone_ASEAN_tuples]
    hour_month_year_zone_nondispatchable_tuples = [(h, m, y, z, te)
                                                   for h in hour for m in month for y in year
                                                   for z in zone for te in nondispatchable_tech]

    ###############################variables#######################################################
    #single-variable
    # cost = model.addVar(vtype=gp.GRB.CONTINUOUS, name='total cost of system [USD]')
    cost_var = model.addVar(vtype=gp.GRB.CONTINUOUS, name='Variable O&M costs [USD]')
    cost_newtech = model.addVar(vtype=gp.GRB.CONTINUOUS, name='Investment costs of new technology [USD]')
    cost_fix = model.addVar(vtype=gp.GRB.CONTINUOUS, name='Fixed O&M costs [USD]')
    cost_newline = model.addVar(vtype=gp.GRB.CONTINUOUS, name='Investment costs of new transmission lines [USD]')
    cost_import = model.addVar(vtype=gp.GRB.CONTINUOUS, name='Electricity import costs from China [USD]')
    #multi-variable
    cap_existing = model.addVars(year, zone, tech,
                                       vtype=gp.GRB.CONTINUOUS, name='Capacity of existing technology [MW]')
    cap_newtech = model.addVars(year, zone, tech,
                                      vtype=gp.GRB.CONTINUOUS, name='Capacity of newbuild technology [MW]')
    cap_newline =  model.addVars(year_zone_zone_tuples,
                                       vtype=gp.GRB.CONTINUOUS, name='Capacity of new transmission lines [MW]')
    cap_lines_existing = model.addVars(year_zone_zone_tuples,
                                             vtype=gp.GRB.CONTINUOUS, name='Capacity of existing transmission line [MW]')
    carbon = model.addVars(year, vtype=gp.GRB.CONTINUOUS, name='Total carbon dioxide emission in each years [Ton]')
    # carbon_capacity = model.addVars(year, zone, vtype=gp.GRB.CONTINUOUS, name='Carbon dioxide emission in each year and each zone [Ton]')
    gen = model.addVars(hour, month, year, zone, tech,
                              vtype=gp.GRB.CONTINUOUS, name='Output of each technology in each year, each zone and each time period [MW]')
    storage = model.addVars(hour_p, month, year, zone, storage_tech,
                                  vtype=gp.GRB.CONTINUOUS, name='Storage in each year, each zone and each time period [MWh]')
    charge = model.addVars(hour, month, year, zone, storage_tech,
                                 vtype=gp.GRB.CONTINUOUS, name='charge in each year, each zone and each time period [MW]')
    trans_export = model.addVars(hour_month_year_zone_zone_tuples,
                                       vtype=gp.GRB.CONTINUOUS, name='Transfer output from zone A to zone B (A is not equals to B)in each year and each time period [MW]')
    trans_import = model.addVars(hour_month_year_zone_zone_tuples,
                                       vtype=gp.GRB.CONTINUOUS, name='Transfer output from zone B to zone A (A is not equals to B)in each year and each time period [MW]')
    remaining_technology = model.addVars(year, zone, tech, vtype=gp.GRB.CONTINUOUS, name='remaining technology [MW]')

    #################################variables  start solution from LP optimization ###################################
    if start==1:
        cost_var.Start = start_variables['cost_var'].X
        cost_newtech.Start = start_variables['cost_newtech'].X
        cost_fix.Start = start_variables['cost_fix'].X
        cost_newline.Start = start_variables['cost_newline'].X
        cap_existing.Start = [start_variables['cap_existing'][y, z, t].x for y in year for z in zone for t in tech]
        cap_newtech.Start = [start_variables['cap_newtech'][y, z, t].x for y in year for z in zone for t in tech]
        cap_newline.Start = [start_variables['cap_newline'][y, z1, z2].X for y in year for z1 in zone for z2 in zone if
                             (y, z1, z2) in year_zone_zone_tuples]
        cap_lines_existing.Start = [start_variables['cap_lines_existing'][y, z1, z2].X for y in year for z1 in zone for
                                    z2 in zone if (y, z1, z2) in year_zone_zone_tuples]
        carbon.Start =[start_variables['carbon'][y].x for y in year]
        gen.Start = [start_variables['gen'][h, m, y, z, t].x for h in hour for m in month for y in year for z in zone
                     for t in tech]
        storage.Start = [start_variables['storage'][h,m,y,z,t].x for h in hour_p for m in month for y in year for z in zone for t in storage_tech]
        charge.Start = [start_variables['charge'][h,m,y,z,t].x for h in hour for m in month for y in year for z in zone for t in storage_tech]
        trans_export.Start = [start_variables['trans_export'][h, m, y, z, z1].x for h in hour for m in month for y in
                              year for z in zone for z1 in zone if (h, m, y, z, z1) in hour_month_year_zone_zone_tuples]
        trans_import.Start = [start_variables['trans_import'][h, m, y, z, z1].x for h in hour for m in month for y in
                              year for z in zone for z1 in zone if (h, m, y, z, z1) in hour_month_year_zone_zone_tuples]
        remaining_technology.Start = [start_variables['remaining_technology'][y, z, t].x for y in year for z in zone for
                                      t in tech]
    ################################# objective funtion: costs ###################################
    model.setObjective(cost_var + cost_newtech + cost_fix + cost_newline + cost_import, gp.GRB.MINIMIZE)

    ################################# constraints ###################################
    ################################### Power balance ###################################
    for z in zone:
        for y in year:
            for m in month:
                for h in hour:
                    imp_z = sum([trans_import[h, m, y, z_out, z]  for z_out in zone if (z != z_out and not np.isnan(para['transmission'][z_out,z]))]) *dt
                    exp_z = sum([trans_export[h, m, y, z, z_in] for z_in in zone if
                             (z != z_in and not np.isnan(para['transmission'][z, z_in]))]) *dt
                    gen_z = sum([gen[h, m, y, z, te] for te in tech]) *dt
                    charge_z = sum([charge[h, m, y, z, te] for te in storage_tech]) *dt
                    demand_z =  para['demand'][z, y, m, h] *dt
                    if (trans_limit != 6) and z=='China':
                        model.addConstr(demand_z == imp_z - exp_z + gen_z/3.0 - charge_z)
                    else:
                        model.addConstr(demand_z == imp_z - exp_z + gen_z - charge_z)
    #################### Transmission capacity constraints #############################
    for y, z_out, z_in in year_zone_zone_tuples:
        remaining_capacity_line = para['transmission'][z_out,z_in]
        new_capacity_line = sum(cap_newline[yy, z_out, z_in] for yy in year[:year.index(y) + 1])
        model.addConstr(cap_lines_existing[y, z_out, z_in] == remaining_capacity_line + new_capacity_line)
    for h,m,y,z_out,z_in in hour_month_year_zone_zone_tuples:
        eff =  para['trans_effi'][z_out, z_in]
        model.addConstr(eff * trans_export[h, m, y, z_out, z_in] == trans_import[h, m, y, z_out, z_in])
    for h,m,y,z_out,z_in in hour_month_year_zone_zone_tuples:
        model.addConstr(trans_export[h, m, y, z_out, z_in] <= cap_lines_existing[y, z_out, z_in])
        if z_in == 'China':
            model.addConstr(trans_export[h, m, y, z_out, z_in] == 0)
    ######################Transmission limits###########################################
    if trans_limit==1: # no new transmission lines
        for y, z_out, z_in in year_zone_zone_tuples:
            model.addConstr(cap_newline[y, z_out, z_in] == 0)
    elif trans_limit==2: # only the planned transmission lines can be built
        for y, z_out, z_in in year_zone_zone_tuples:
            new_capacity_line = sum(cap_newline[yy, z_out, z_in] for yy in year[:year.index(y) + 1])
            new_capacity_limit = para['transmission limit'][z_out,z_in]
            model.addConstr(new_capacity_line <= new_capacity_limit)
    elif trans_limit ==3: # the Mekong coutries' transmission lines can be extended except China. self-sucifficient--1/3 demand can be imported （non-used scenario)
        for z in zone:
            for y in year:
                imp_y = sum(trans_import[h,m,y,z_out,z] for h in hour for m in month for z_out in zone if (z != z_out and not np.isnan(para['transmission'][z_out,z])))*dt/weight
                demand_y = sum(para['demand'][z,y,m,h] for h in hour for m in month)*dt/weight
                if z!='China':
                    model.addConstr(imp_y <=  demand_y/3.0 )
        for y in year:
            for z in ['Laos','Myanmar','Vietnam']:
                new_capacity_line_China = sum(cap_newline[yy, 'China', z] for yy in year[:year.index(y) + 1])
                new_capacity_limit = para['transmission limit']['China', z]
                model.addConstr(new_capacity_line_China <= new_capacity_limit)
    elif trans_limit==4:#China export constraint relax
        for z in zone:
            for y in year:
                imp_y = sum(trans_import[h,m,y,z_out,z] for h in hour for m in month for z_out in zone if (z != z_out and not np.isnan(para['transmission'][z_out,z])))*dt/weight
                demand_y = sum(para['demand'][z,y,m,h] for h in hour for m in month)*dt/weight
                if z!='China':
                    model.addConstr(imp_y <=  demand_y/3.0 )

    ############################  Maximum output constraint ############################
    model.addConstrs(gen[h, m, y, z, te] <= cap_existing[y, z, te]
                     for h in hour for m in month for y in year for z in zone for te in tech)

    ############################  Upper bound of  investment capacity ############################
    model.addConstrs(cap_existing[y, z, te] <= para['tech_upper'][te, z] for y in year for z in zone for te in tech if para['tech_upper'][te, z] != np.Inf)
    model.addConstrs(cap_newtech[y, z, te] <= para['newtech_upper'][te, z] for y in year for z in zone for te in tech if para['newtech_upper'][te, z] != np.Inf)
    model.addConstrs(cap_newtech[y, z, te] >= para['newtech_lower'][te, z] for y in year for z in zone for te in tech if  para['newtech_lower'][te, z]>0)

    ############################  nondispatchable energy output ############################
    if nondispatchable_tech != 0:
        model.addConstrs(gen[h, m, y, z, te] <= para['capacity_factor'][te, z, m, h] * cap_existing[y, z, te]
                         for h,m,y,z,te in hour_month_year_zone_nondispatchable_tuples)
    ##################################  Lifetime ####################################
    for y in year:
        for z in zone:
            for te in tech:
                lifetime = para['lifetime'][te,y]
                service_time = y - 2020          #para['year_sets'][0]
                remaining_time_ideal = int(lifetime - service_time)
                if remaining_time_ideal <= 1:
                    model.addConstr(remaining_technology[y, z, te]  == 0)
                else :
                    model.addConstr(remaining_technology[y, z, te] == sum([para['age_'][z, te, a] for a in range(1, remaining_time_ideal)]))
    ##################################  Storage ####################################
    if storage_tech != 0:
        model.addConstrs(storage[h, m, y, z, te] == storage[h-1, m, y, z, te] - gen[h, m, y, z, te] * dt + charge[h, m, y, z, te]* dt * para['efficiency'][te, y]
                         for h in hour for m in month for y in year for z in zone for te in storage_tech)  #balance
        model.addConstrs(storage[0, m, y, z, te] == para['storage_level'][te, z] * cap_existing[y, z, te] * para['storage_ratio'][te, z]
                         for m in month for y in year for z in zone for te in storage_tech)  #initial condition
        model.addConstrs(storage[hour_p[-1], m, y, z, te] == storage[0, m, y, z, te]
                         for m in month for y in year for z in zone for te in storage_tech)  #end condition
        model.addConstrs(storage[h, m, y, z, te] <= cap_existing[y, z, te] * para['storage_ratio'][te, z]
                         for h in hour for m in month for y in year for z in zone for te in storage_tech) #maximum electricity storage volumne constraint
        # model.addConstrs(gen[h, m, y, z, te] * dt <= storage[h-1, m, y, z, te]
        #                  for h in hour for m in month for y in year for z in zone for te in storage_tech)
    ##################################  Ramping ####################################
    model.addConstrs(gen[h, m, y, z, te] - gen[h-1, m, y, z, te] <= para['ramp_up'][te] * dt * cap_existing[y, z, te]
                         for h in hour for m in month for y in year for z in zone for te in dispatchable_tech if (h>1 and (para['ramp_up'][te] * dt)<1))
    model.addConstrs(gen[h-1, m, y, z, te] - gen[h, m, y, z, te] <= para['ramp_down'][te] * dt * cap_existing[y, z, te]
                         for h in hour for m in month for y in year for z in zone for te in dispatchable_tech if (h>1 and (para['ramp_down'][te] * dt)<1))

    ################################# constraints: variable costs ###################################
    var_OM_cost_cost = sum([para['varcost'][te, y] * gen[h, m, y, z, te] * dt *  para['var_factor'][y]
                            for h in hour for m in month for y in year for z in zone_ASEAN for te in tech]) / weight
    fuel_cost = sum([para['fuelprice'][te, y] * gen[h, m, y, z, te] * dt * para['var_factor'][y]
                     for h in hour for m in month for y in year for z in zone_ASEAN for te in tech]) / weight
    var_OM_line_cost = 0.5 * sum([para['varcost_lines'][z, z1] * trans_export[h, m, y, z, z1] * dt * para['var_factor'][y]
                                        for h, m, y, z, z1 in hour_month_year_zone_zone_ASEAN_tuples]) / weight
    cost_import = sum(elec_price * trans_export[h, m, y, 'China', z] * dt * para['var_factor'][y]
                      for h in hour for m in month for y in year for z in ['Myanmar','Laos','Vietnam'])/weight
    # variable cost
    model.addConstr(cost_var == var_OM_cost_cost + fuel_cost + var_OM_line_cost + cost_import)
    # new built cost
    model.addConstr(cost_newtech == sum(para['invcost'][te, y] * cap_newtech[y, z, te] * \
                              para['inv_factor'][te, y]
                              for y in year for z in zone_ASEAN for te in tech))
    model.addConstr(cost_newline == 0.5 * sum(para['invline'][z, z1] * cap_newline[y, z, z1] * \
                                               para['distance'][z, z1] * \
                                               para['trans_inv_factor'][y]
                                               for y,z,z1 in year_zone_zone_ASEAN_tuples))
    # fix cost
    fix_cost_tech = sum(para['fixcost'][te,y] * cap_existing[y, z, te]
                                     * para['fix_factor'][y]
                                     for y in year for z in zone_ASEAN for te in tech)
    fix_cost_line = 0.5 * sum(para['fixcost_lines'][z, z1] * cap_lines_existing[y, z, z1]
                              * para['fix_factor'][y]
                              for y,z,z1 in year_zone_zone_ASEAN_tuples)
    model.addConstr(cost_fix == fix_cost_tech+ fix_cost_line)

    ################################# constraints: existing capacity ###################################
    for y in year:
        for z in zone:
            for te in tech:
                new_tech = sum([cap_newtech[yy, z, te] for yy in year[:year.index(y) + 1]
                                if y - yy  < para['lifetime'][te, yy]])
                model.addConstr(cap_existing[y, z, te] == remaining_technology[y, z, te] + new_tech)
    model.addConstrs(cap_newline[y, z, z1] == cap_newline[y,z1,z] for y,z,z1 in year_zone_zone_tuples)  # new constraint
    # model.addConstrs(cap_newline[y, z, z1] ==0 for y,z, z1 in year_zone_zone_tuples if y==2020)
    ################################### Carbon dioxide emission ###################################
    model.addConstrs(carbon[y] <= float(para['carbon_limit'][y]) for y in year if float(para['carbon_limit'][y]) != np.Inf)
    model.addConstrs(carbon[y] == sum(para['carbon'][te, y] * gen[h, m, y, z, te] * dt for h in hour for m in month for te in dispatchable_tech for z in zone)/weight for y in year)
    # model.addConstrs(carbon_capacity[y,z] == sum(para['carbon'][te, y] * gen[h, m, y, z, te] * dt for h in hour for m in month for te in tech) for z in zone)

    #  add hydro
    if ishydro ==1:  # optimize the hydro construction and operation
        if start==0:  #without start solution
            model, result = add_hydro_1(model, para, dt, gen, cap_newtech, weight, result, Sed_Cons, start=start)
        elif start==1: # with start solution containing all variables
            model,result = add_hydro_1(model, para, dt, gen, cap_newtech, weight, result, Sed_Cons, start=start, start_variables=start_variables)
        elif start==2:  # with start solution containing only binary variables
            model,result = add_hydro_1(model, para, dt, gen, cap_newtech, weight, result, Sed_Cons, start=start, start_variables=start_variables)
    elif ishydro==2:  # only optimize the hydro operation, need hydro potfolio
        # print('ishydro==========',ishydro)
        model,result = add_hydro_2(model,para,dt,gen,cap_newtech, weight, result, Sed_Cons, hydro_portfolio)
    else:
        model.addConstrs(gen[h, m, y, z, [i for i,j  in para['type'].items() if j == 'hydro'][0]]  == float(para['hydro_output']['Hydro', z, m, h] )
                         for h in hour for m in month for y in year  for z in zone)

    # parameter result
    result['year']= year
    result['zone']=zone
    result['tech']=tech
    result['hour']=hour
    result['hour_p']=hour_p
    result['month']=month
    result['tech_storage'] = storage_tech
    result['year_zone_zone_tuples'] = year_zone_zone_tuples
    result['hour_month_year_zone_zone_tuples'] = hour_month_year_zone_zone_tuples
    result['hour_month_year_zone_nondispatchable_tuples'] = hour_month_year_zone_nondispatchable_tuples
    # variable result
    result['cost_var'] = cost_var
    result['cost_newtech'] = cost_newtech
    result['cost_fix'] = cost_fix
    result['cost_newline'] = cost_newline
    result['carbon'] = carbon
    result['cap_existing'] = cap_existing
    result['cap_newtech'] = cap_newtech
    result['cap_newline'] = cap_newline
    result['cap_lines_existing'] = cap_lines_existing
    result['gen'] = gen
    result['storage'] = storage
    result['charge'] = charge
    result['trans_export'] = trans_export
    result['trans_import'] = trans_import
    result['remaining_technology'] = remaining_technology
    # output result
    result['var_OM_cost_cost'] = var_OM_cost_cost
    result['fuel_cost'] = fuel_cost
    result['var_OM_line_cost'] = var_OM_line_cost
    result['fix_cost_tech'] = fix_cost_tech
    result['fix_cost_line'] = fix_cost_line


    return model,result



def add_hydro_1(model, para, dt, gen, cap_newtech, weight, result, Sed_Cons, start=0, start_variables=[]):
    # sets
    year = para['year_sets']
    zone = para['zone_sets']
    tech = para['tech_sets']
    hour = para['hour_sets']
    hour_p = [0]+para['hour_sets']
    month = para['month_sets']
    station = para['stcd_sets']
    station_season=para['stcd_sets_season']
    station_daily=para['stcd_sets_daily']
    station_runoff=para['stcd_sets_runoff']
    station_unbuilt = para['stcd_sets_unbuilt']
    station_hour_month_year_tuples = [(s, h, m, y) for s in station for h in hour for m in month for y in year]
    station_month_year_tuples = [(s,m,y) for s in station for m in month for y in year]
    station_hour_p_month_year_tuples = [(s,h,m,y) for s in station for h in hour_p for m in month for y in year]

    ################################## hydropower operation start #########################
    # Hydropower plant variables
    if len(station_unbuilt)>0:
        isbuilt = model.addVars(station_unbuilt, vtype=gp.GRB.BINARY, name='if built dams 0-1')
        if start==1:
            isbuilt.Start = [start_variables['isbuilt'][s] for s in station_unbuilt]
        if start==2:
            isbuilt.Start = start_variables
    else:
        isbuilt = model.addVars(station, vtype=gp.GRB.BINARY, name='if built dams 0-1')

    naturalinflow = model.addVars(station_hour_month_year_tuples, lb=-np.Inf, ub =np.inf,
                                       vtype=gp.GRB.CONTINUOUS,
                                       name='natural inflow of reservoir [m3/s]')
    inflow = model.addVars(station_hour_month_year_tuples, lb=-np.Inf, ub =np.inf,
                                 vtype=gp.GRB.CONTINUOUS,name='inflow of reservoir [m3/s]')
    outflow = model.addVars(station_hour_month_year_tuples,
                                  vtype=gp.GRB.CONTINUOUS,name='outflow of reservoir [m3/s]')
    genflow = model.addVars(station_hour_month_year_tuples,
                                  vtype=gp.GRB.CONTINUOUS,name='generation flow of reservoir [m3/s]')
    spillflow = model.addVars(station_hour_month_year_tuples,
                                    vtype=gp.GRB.CONTINUOUS,name='water spillage flow of reservoir [m3/s]')
    storage_hydro =model.addVars(station_hour_p_month_year_tuples,
                                       vtype=gp.GRB.CONTINUOUS,name='storage of reservoir [10^8 m3]')
    output = model.addVars(station_hour_month_year_tuples, vtype=gp.GRB.CONTINUOUS,
                       name='output of reservoir [MW]')
    if start==1:
        naturalinflow.Start = [start_variables['naturalinflow'][s,h,m,y].x for s in station for h in hour for m in month for y in year ]
        inflow.Start = [start_variables['inflow'][s, h, m, y].x for s in station for h in hour for m in
                               month for y in year]
        outflow.Start = [start_variables['outflow'][s, h, m, y].x for s in station for h in hour for m in
                        month for y in year]
        genflow.Start = [start_variables['genflow'][s, h, m, y].x for s in station for h in hour for m in
                         month for y in year]
        spillflow.Start = [start_variables['spillflow'][s, h, m, y].x for s in station for h in hour for m in
                         month for y in year]
        storage_hydro.Start = [start_variables['storage_hydro'][s, h, m, y].x for s in station for h in hour_p for m in
                           month for y in year]
        output.Start = [start_variables['output'][s, h, m, y].x for s in station for h in hour for m in
                           month for y in year]
    ################################constraints##########################################
   ################################# Hydropower output ###################################
    # inflow
    model.addConstrs(naturalinflow[s, h, m, y] == para['inflow'][s, m, h]
                     for s in station for h in hour for m in month for y in year)
    # streamflow from upstream
    for s in station:
        for y in year:
            for m in month:
                for h in hour:
                    up_stream_outflow = 0
                    for ups, delay in zip(para['connect'][para['connect']['NEXTPOWER_ID'] == s].POWER_ID, para['connect'][para['connect']['NEXTPOWER_ID']==s].delay):
                        delay_dt = int(int(delay)/dt)
                        while delay_dt> hour[-1]:
                            delay_dt = delay_dt- hour[-1]
                        if (h - delay_dt >= hour[0]):
                            up_stream_outflow += outflow[ups, h - delay_dt, m, y]
                        else:
                            up_stream_outflow += outflow[ups, hour[-1] - delay_dt + h, m, y]
                    model.addConstr(inflow[s, h, m, y] == naturalinflow[s, h, m, y] + up_stream_outflow)
    #water balance
    model.addConstrs(storage_hydro[s, h, m, y] == storage_hydro[s, h-1, m, y] +  (inflow[s, h, m, y]-outflow[s, h, m, y])*3600*dt*1e-8
                     for s in station for h in hour for m in month for y in year)
    #discharge rule
    model.addConstrs(outflow[s, h, m, y] == genflow[s, h, m, y] + spillflow[s, h, m, y]
                     for s in station for h in hour for m in month for y in year)
    #outflow low bound
    model.addConstrs(outflow[s, h, m, y] >= para['static']['outflow_min', s]
                     for s in station for h in hour for m in month for y in year if para['static']['outflow_min', s]>0)
    #outflow up bound
    # model.addConstrs(outflow[s, h, m, y] <= para['static']['outflow_max', s]
    #                  for s in station for h in hour for m in month for y in year if para['static']['outflow_max', s]!=np.Inf)
    #storage low bound
    model.addConstrs(storage_hydro[s, h, m, y] >= para['storagedown'][s, m, h]
                     for s in station for h in hour for m in month for y in year)
    model.addConstrs(storage_hydro[s, 0, m, y] >= para['storagedown'][s, m, 1]
                     for s in station for m in month for y in year)
    #storage up bound
    model.addConstrs(storage_hydro[s, h, m, y] <= para['storageup'][s, m, h]
                     for s in station for h in hour for m in month for y in year)
    model.addConstrs(storage_hydro[s, 0, m, y] <= para['storageup'][s, m, 1]
                     for s in station for m in month for y in year)
    # output low bound
    model.addConstrs(output[s, h, m, y] >= para['static']['N_min', s]
                     for s in station for h in hour for m in month for y in year if para['static']['N_min', s]>0)
    #output up bound
    model.addConstrs(output[s, h, m, y] <= para['static']['N_max', s]
                     for s in station for h in hour for m in month for y in year)
    #output calculation
    model.addConstrs(output[s, h, m, y] == para['static']['coeff', s] * genflow[s, h, m, y] * para['static']['head', s] * 1e-3  # Kw to Mw
                     for s in station for h in hour for m in month for y in year)
    #initial and end storage
    if len(month)>1: #non-continuous
        for m in month:
            # daily-operation stations, given the initial and end storage in a day
            model.addConstrs(storage_hydro[s, hour_p[0], m, y] ==  storage_hydro[s, hour_p[-1], m, y]
                             for s in station_daily for y in year)
            # runoff stations
            model.addConstrs(outflow[s, h, m, y] == inflow[s, h, m, y]
                             for s in station_runoff for h in hour for y in year)
            # season-operation stations
            if m!=month[-1]:
                model.addConstrs(storage_hydro[s, hour_p[-1], m, y] == (
                        storage_hydro[s, hour_p[0], m + 1, y] - storage_hydro[s, hour_p[0], m, y]) * weight +
                                 storage_hydro[s, hour_p[0], m, y]
                                 for s in station_season for y in year)
            elif m==month[-1]: # the last month
                model.addConstrs(storage_hydro[s, hour_p[-1], m, y] == (
                        storage_hydro[s, hour_p[0], month[0], y] - storage_hydro[s, hour_p[0], m, y]) * weight +
                                 storage_hydro[s, hour_p[0], m, y]
                                 for s in station_season for y in year)
    else:# continuous
        model.addConstrs(storage_hydro[s, hour_p[0], m, y] == para['storageinit'][m,s]
                     for s in station for m in month for y in year)
        model.addConstrs(storage_hydro[s, hour_p[-1], m, y] == para['storageend'][m,s]
                     for s in station for m in month for y in year)
    # hydropower statistic in a zone
    for z in zone:
        for y in year:
            for m in month:
                for h in hour:
                    hydro_output = 0
                    for s in station:
                        if para['static']['zone',s] == z:
                            hydro_output += output[s, h, m, y] * dt
                    model.addConstr(gen[h, m, y, z, 'Hydro'] == hydro_output)

    # dynammic hydropower built
    if len(station_unbuilt) > 0:
        model.addConstrs(spillflow[s,h,m,y] == outflow[s,h,m,y]
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) > y  )
        model.addConstrs(outflow[s,h,m,y] == inflow[s,h,m,y]
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) > y )
        model.addConstrs(spillflow[s,h,m,y] <= outflow[s,h,m,y] + isbuilt[s]*para['static']['outflow_max',s]*10
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) <= y)
        model.addConstrs(spillflow[s,h,m,y] >= outflow[s,h,m,y] - isbuilt[s]*para['static']['outflow_max',s]*10
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) <= y)
        model.addConstrs(outflow[s,h,m,y] <= inflow[s,h,m,y] + isbuilt[s]*para['static']['outflow_max',s]*10
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) <= y)
        model.addConstrs(outflow[s,h,m,y] >= inflow[s,h,m,y] - isbuilt[s]*para['static']['outflow_max',s]*10
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) <= y)
    # hydropower new built cost
    for z in zone:
        for y in year:
            if y==2030:
                new_built_hydro = 0
                for s in station_unbuilt:
                    if para['static']['zone',s] == z:
                        new_built_hydro += para['static']['N',s] * isbuilt[s]
                model.addConstr(cap_newtech[2030, z, 'Hydro'] == new_built_hydro)
            else:
                model.addConstr(cap_newtech[y, z, 'Hydro'] == 0)
    # add sed constraint
    if Sed_Cons>0:
        #########################parameters#################################
        basins = para['basins']
        dams = para['dams']
        basin = basins.FROM_NODE.values
        dam = dams.SUB.values
        sed_yield = {k: v for k, v in zip(basins.FROM_NODE.values, basins.Sed_Yield.values)}
        dep = {k: v for k, v in zip(basins.FROM_NODE.values, basins.Sed_Dep.values)}
        Te = {k: dams[dams.SUB == k].TE.values[0] for k in dams.SUB.values}
        ########################Variables#####################
        out_sed = model.addVars(basin,vtype=gp.GRB.CONTINUOUS, name='output basin sediment to down stream [MT]')
        acc_sed = model.addVars(basin,  vtype=gp.GRB.CONTINUOUS, name='accumulation basin sediment [MT]')
        if start==1:
            out_sed.Start = [start_variables['out_sed'][b].x for b in basin]
            acc_sed.Start = [start_variables['acc_sed'][b].x for b in basin]
        # sed_total = model.addVar(vtype=gp.GRB.CONTINUOUS, name='total accumulate sediment to delta [MT]')
        #####################constraints######################
        for b in basin:
            if len(basins[basins.TO_NODE == b].FROM_NODE.values) > 0:
                model.addConstr(acc_sed[b] == sum([out_sed[n] for n in basins[basins.TO_NODE == b].FROM_NODE.values]) + sed_yield[b])
            else:
                model.addConstr(acc_sed[b] == sed_yield[b])
        M=100000
        for b in basin:
            if b in dam:
                if b in station_unbuilt:
                    if dep[b]==Te[b]:
                        model.addConstr(out_sed[b] == acc_sed[b] * (1 - dep[b]))
                    else:
                        # model.addConstr(out_sed[b] == acc_sed[b] * (1 - dep[b] - isbuilt[b]*(Te[b] - dep[b])))
                        model.addConstr( (out_sed[b] - acc_sed[b] * (1 - dep[b]))/(dep[b] - Te[b])   <=  acc_sed[b] + (1-isbuilt[b])*M)
                        model.addConstr( (out_sed[b] - acc_sed[b] * (1 - dep[b]))/(dep[b] - Te[b])   >=  acc_sed[b] - (1-isbuilt[b])*M)
                        model.addConstr( (out_sed[b] - acc_sed[b] * (1 - dep[b]))/(dep[b] - Te[b])   <=  isbuilt[b]*M)
                        model.addConstr( (out_sed[b] - acc_sed[b] * (1 - dep[b]))/(dep[b] - Te[b])   >=  -isbuilt[b] * M)
                else:
                    model.addConstr(out_sed[b] == acc_sed[b] * (1 - Te[b]) )
            else:
                model.addConstr(out_sed[b] == acc_sed[b] * (1-dep[b]))
        # model.addConstr(sed_total == out_sed[540])
        model.addConstr(out_sed[540] >= Sed_Cons)
        result['sediment'] = out_sed[540]
        result['out_sed']=out_sed
        result['acc_sed']=acc_sed



    result['naturalinflow'] = naturalinflow
    result['inflow'] = inflow
    result['outflow'] = outflow
    result['genflow'] = genflow
    result['spillflow'] = spillflow
    result['storage_hydro'] = storage_hydro
    result['output'] = output
    result['sed_con'] = Sed_Cons
    result['isbuilt'] = isbuilt
    result['station_unbuilt']=station_unbuilt
    result['station'] = station
    result['station_hour_month_year_tuples'] = station_hour_month_year_tuples
    result['station_hour_p_month_year_tuples'] = station_hour_p_month_year_tuples
    return model,result



def add_hydro_2(model, para, dt, gen, cap_newtech, weight, result, Sed_Cons, hydro_potfolio):
    # sets
    year = para['year_sets']
    zone = para['zone_sets']
    # tech = para['tech_sets']
    hour = para['hour_sets']
    hour_p = [0]+para['hour_sets']
    month = para['month_sets']
    station = para['stcd_sets']
    station_season = para['stcd_sets_season']
    station_daily = para['stcd_sets_daily']
    station_runoff = para['stcd_sets_runoff']
    station_unbuilt = para['stcd_sets_unbuilt']
    station_hour_month_year_tuples = [(s, h, m, y) for s in station for h in hour for m in month for y in year]
    # station_month_year_tuples = [(s,m,y) for s in station for m in month for y in year]
    station_hour_p_month_year_tuples = [(s,h,m,y) for s in station for h in hour_p for m in month for y in year]

    ################################## hydropower operation start #########################
    # Hydropower plant variables
    isbuilt = dict(zip(station_unbuilt,hydro_potfolio ))
    naturalinflow = model.addVars(station_hour_month_year_tuples, lb=-np.Inf, ub =np.inf,
                                       vtype=gp.GRB.CONTINUOUS,
                                       name='natural inflow of reservoir [m3/s]')
    inflow = model.addVars(station_hour_month_year_tuples, lb=-np.Inf, ub =np.inf,
                                 vtype=gp.GRB.CONTINUOUS,name='inflow of reservoir [m3/s]')
    outflow = model.addVars(station_hour_month_year_tuples,
                                  vtype=gp.GRB.CONTINUOUS,name='outflow of reservoir [m3/s]')
    genflow = model.addVars(station_hour_month_year_tuples,
                                  vtype=gp.GRB.CONTINUOUS,name='generation flow of reservoir [m3/s]')
    spillflow = model.addVars(station_hour_month_year_tuples,
                                    vtype=gp.GRB.CONTINUOUS,name='water spillage flow of reservoir [m3/s]')
    storage_hydro =model.addVars(station_hour_p_month_year_tuples,
                                       vtype=gp.GRB.CONTINUOUS,name='storage of reservoir [10^8 m3]')
    output = model.addVars(station_hour_month_year_tuples, vtype=gp.GRB.CONTINUOUS,
                       name='output of reservoir [MW]')



    ################################# Hydropower output ###################################
    # inflow
    model.addConstrs(naturalinflow[s, h, m, y] == para['inflow'][s, m, h]
                     for s in station for h in hour for m in month for y in year)
    # streamflow from upstream
    for s in station:
        for y in year:
            for m in month:
                for h in hour:
                    up_stream_outflow = 0
                    for ups, delay in zip(para['connect'][para['connect']['NEXTPOWER_ID'] == s].POWER_ID, para['connect'][para['connect']['NEXTPOWER_ID']==s].delay):
                        delay_dt = int(int(delay)/dt)
                        while delay_dt> hour[-1]:
                            delay_dt = delay_dt- hour[-1]
                        if (h - delay_dt >= hour[0]):
                            up_stream_outflow += outflow[ups, h - delay_dt, m, y]
                        else:
                            up_stream_outflow += outflow[ups, hour[-1] - delay_dt + h, m, y]
                    model.addConstr(inflow[s, h, m, y] == naturalinflow[s, h, m, y] + up_stream_outflow)
    #water balance
    model.addConstrs(storage_hydro[s, h, m, y] == storage_hydro[s, h-1, m, y] +  (inflow[s, h, m, y]-outflow[s, h, m, y])*3600*dt*1e-8
                     for s in station for h in hour for m in month for y in year)
    #discharge rule
    model.addConstrs(outflow[s, h, m, y] == genflow[s, h, m, y] + spillflow[s, h, m, y]
                     for s in station for h in hour for m in month for y in year)

    #low bound
    model.addConstrs(outflow[s, h, m, y] >= para['static']['outflow_min', s]
                     for s in station for h in hour for m in month for y in year if para['static']['outflow_min', s]>0)
    #up bound
    # model.addConstrs(outflow[s, h, m, y] <= para['static']['outflow_max', s]
    #                  for s in station for h in hour for m in month for y in year if para['static']['outflow_max', s]!=np.Inf)

    #storage low bound
    model.addConstrs(storage_hydro[s, h, m, y] >= para['storagedown'][s, m, h]
                     for s in station for h in hour for m in month for y in year)
    model.addConstrs(storage_hydro[s, 0, m, y] >= para['storagedown'][s, m, 1]
                     for s in station for m in month for y in year)
    #storage up bound
    model.addConstrs(storage_hydro[s, h, m, y] <= para['storageup'][s, m, h]
                     for s in station for h in hour for m in month for y in year)
    model.addConstrs(storage_hydro[s, 0, m, y] <= para['storageup'][s, m, 1]
                     for s in station for m in month for y in year)
    # output low bound
    model.addConstrs(output[s, h, m, y] >= para['static']['N_min', s]
                     for s in station for h in hour for m in month for y in year if para['static']['N_min', s]>0)
    #output up bound
    model.addConstrs(output[s, h, m, y] <= para['static']['N_max', s]
                     for s in station for h in hour for m in month for y in year)
    #output calculation
    model.addConstrs(output[s, h, m, y] == para['static']['coeff', s] * genflow[s, h, m, y] * para['static']['head', s] * 1e-3  # Kw to Mw
                     for s in station for h in hour for m in month for y in year)
    #initial and end storage
    if len(month)>1: #non-continuous
        for m in month:
            model.addConstrs(storage_hydro[s, hour_p[0], m, y] ==  storage_hydro[s, hour_p[-1], m, y]
                             for s in station_daily for y in year)
            # runoff stations
            model.addConstrs(outflow[s, h, m, y] == inflow[s, h, m, y]
                             for s in station_runoff for h in hour for y in year)
            # season-operation stations
            if m!=month[-1]:
                model.addConstrs(storage_hydro[s, hour_p[-1], m, y] == (
                        storage_hydro[s, hour_p[0], m + 1, y] - storage_hydro[s, hour_p[0], m, y]) * weight +
                                 storage_hydro[s, hour_p[0], m, y]
                                 for s in station_season for y in year)
            elif m==month[-1]: # the last month
                model.addConstrs(storage_hydro[s, hour_p[-1], m, y] == (
                        storage_hydro[s, hour_p[0], month[0], y] - storage_hydro[s, hour_p[0], m, y]) * weight +
                                 storage_hydro[s, hour_p[0], m, y]
                                 for s in station_season for y in year)
    else:# continuous
        model.addConstrs(storage_hydro[s, hour_p[0], m, y] == para['storageinit'][m,s]
                     for s in station for m in month for y in year)
        model.addConstrs(storage_hydro[s, hour_p[-1], m, y] == para['storageend'][m,s]
                     for s in station for m in month for y in year)


    # hydropower statistic in a zone
    for z in zone:
        for y in year:
            for m in month:
                for h in hour:
                    hydro_output = 0
                    for s in station:
                        if para['static']['zone',s] == z:
                            hydro_output += output[s, h, m, y] * dt
                    model.addConstr(gen[h, m, y, z, 'Hydro'] == hydro_output)

    # dynammic hydropower built according to isbuilt
    model.addConstrs(spillflow[s,h,m,y] == outflow[s,h,m,y]
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) > y  )
    model.addConstrs(outflow[s,h,m,y] == inflow[s,h,m,y]
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) > y )
    model.addConstrs(spillflow[s,h,m,y] == outflow[s,h,m,y]
                     for s in station_unbuilt for h in hour for m in month for y in year if isbuilt[s] ==0 )
    model.addConstrs(outflow[s,h,m,y] == inflow[s,h,m,y]
                     for s in station_unbuilt for h in hour for m in month for y in year if isbuilt[s] ==0 )

    model.addConstrs(spillflow[s,h,m,y] <= outflow[s,h,m,y] + isbuilt[s]*para['static']['outflow_max',s]*10
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) <= y)
    model.addConstrs(spillflow[s,h,m,y] >= outflow[s,h,m,y] - isbuilt[s]*para['static']['outflow_max',s]*10
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) <= y)

    model.addConstrs(outflow[s,h,m,y] <= inflow[s,h,m,y] + isbuilt[s]*para['static']['outflow_max',s]*10
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) <= y)
    model.addConstrs(outflow[s,h,m,y] >= inflow[s,h,m,y] - isbuilt[s]*para['static']['outflow_max',s]*10
                     for s in station_unbuilt for h in hour for m in month for y in year if int(para['static']['COD',s]) <= y)
    # hydropower new built cost
    for z in zone:
        for y in year:
            if y==2030:
                new_built_hydro = 0
                for s in station_unbuilt:
                    if para['static']['zone',s] == z:
                        new_built_hydro += para['static']['N',s] * isbuilt[s]
                model.addConstr(cap_newtech[2030, z, 'Hydro'] == new_built_hydro)
            else:
                model.addConstr(cap_newtech[y, z, 'Hydro'] == 0)
    # calculate sediment constraint
    if Sed_Cons>0:

        #########################parameters#################################
        basins = para['basins']
        dams = para['dams']
        basin = basins.FROM_NODE.values
        dam = dams.SUB.values
        sed_yield = {k: v for k, v in zip(basins.FROM_NODE.values, basins.Sed_Yield.values)}
        dep = {k: v for k, v in zip(basins.FROM_NODE.values, basins.Sed_Dep.values)}
        Te = {k: dams[dams.SUB == k].TE.values[0] for k in dams.SUB.values}
        ########################Variables#####################
        out_sed = model.addVars(basin,vtype=gp.GRB.CONTINUOUS, name='output basin sediment to down stream [MT]')
        acc_sed = model.addVars(basin,  vtype=gp.GRB.CONTINUOUS, name='accumulation basin sediment [MT]')
        # sed_total = model.addVar(vtype=gp.GRB.CONTINUOUS, name='total accumulate sediment to delta [MT]')
        #####################constraints######################
        for b in basin:
            if len(basins[basins.TO_NODE == b].FROM_NODE.values) > 0:
                model.addConstr(acc_sed[b] == sum([out_sed[n] for n in basins[basins.TO_NODE == b].FROM_NODE.values]) + sed_yield[b])
            else:
                model.addConstr(acc_sed[b] == sed_yield[b])
        # M=100000
        for b in basin:
            if b in dam:
                if b in station_unbuilt:
                    if dep[b]==Te[b]:
                        model.addConstr(out_sed[b] == acc_sed[b] * (1 - dep[b]))
                    else:
                        model.addConstr(out_sed[b] == acc_sed[b] * (1 - dep[b] - isbuilt[b]*(Te[b] - dep[b])))
                        # model.addConstr( (out_sed[b] - acc_sed[b] * (1 - dep[b]))/(dep[b] - Te[b])   <=  acc_sed[b] + (1-isbuilt[b])*M)
                        # model.addConstr( (out_sed[b] - acc_sed[b] * (1 - dep[b]))/(dep[b] - Te[b])   >=  acc_sed[b] - (1-isbuilt[b])*M)
                        # model.addConstr( (out_sed[b] - acc_sed[b] * (1 - dep[b]))/(dep[b] - Te[b])   <=  isbuilt[b]*M)
                        # model.addConstr( (out_sed[b] - acc_sed[b] * (1 - dep[b]))/(dep[b] - Te[b])   >=  -isbuilt[b] * M)
                else:
                    model.addConstr(out_sed[b] == acc_sed[b] * (1 - Te[b]) )
            else:
                model.addConstr(out_sed[b] == acc_sed[b] * (1-dep[b]))
        # model.addConstr(sed_total == out_sed[540])
        # model.addConstr(out_sed[540] >= Sed_Cons)
        result['sediment'] = out_sed[540]
        result['out_sed']=out_sed
        result['acc_sed']=acc_sed





    result['naturalinflow'] = naturalinflow
    result['inflow'] = inflow
    result['outflow'] = outflow
    result['genflow'] = genflow
    result['spillflow'] = spillflow
    result['storage_hydro'] = storage_hydro
    result['output'] = output
    result['sed_con'] = Sed_Cons
    result['isbuilt'] = isbuilt
    result['station_unbuilt']=station_unbuilt
    result['station'] = station
    result['station_hour_month_year_tuples'] = station_hour_month_year_tuples
    result['station_hour_p_month_year_tuples'] = station_hour_p_month_year_tuples
    return model,result








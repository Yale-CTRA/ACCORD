"""
This is a simple script to preprocess the data into a static, one-row-per-patient format
This file should be run before any anaylysis is performed since it creates the dataset
Created by: Aditya Biswas
"""



import numpy as np
import pandas as pd
import sys
path = '/home/aditya/Projects/'
sys.path.append(path)
from Helper.containers import Data
from Helper.utilities import save
import os
path = '/home/aditya/Projects/ACCORD/Data/Original Data/'
os.chdir(path)


## load every ACCORD file into dictionary with lowercase name as key (converted sas -> csv beforehand)
dataDict = {}
contents = os.listdir()
for file in contents:
    if file[-4:] == '.csv':
        dataDict[file[:-4].lower()] = pd.read_csv(file, index_col = 0)

####################################################################################################################################
subSet = lambda data: data['Visit'] == 'BLR'  # returns boolean index for all first patient records

# treatment markers
m = len(dataDict['accord_key'])
actions = np.empty((m, 3))
actions[:,:] = np.nan
index = dataDict['accord_key'].columns.get_loc('treatment')
for i in range(m):
    string = dataDict['accord_key'].iloc[i,index]
    slashLoc = string.index('/')
    firstString, secondstring = string[:slashLoc], string[slashLoc+1:]
    if firstString == 'Standard Glycemia' or firstString == 'Standard Gylcemia':
        actions[i,0] = 0
    elif firstString == 'Intensive Glycemia' or firstString == 'Intensive Gylcemia':
        actions[i,0] = 1
    if secondstring == 'Standard BP':
        actions[i,1] = 0
    elif secondstring == 'Intensive BP':
        actions[i,1] = 1
    if secondstring == 'Lipid Placebo':
        actions[i,2] = 0
    elif secondstring == 'Lipid Fibrate':
        actions[i,2] = 1
actions = pd.DataFrame(actions, index = dataDict['accord_key'].index, columns = ['glycemia', 'bp', 'lipid'])
actions.fillna(value = 0, inplace = True)           # those not assigned to a trial get a 0 b/c effectively equivalent to controls


# demographics
dataDict['accord_key'].rename({'baseline_age': 'age', 'raceclass2': 'white'}, axis = 1, inplace = True)
demoVars = dataDict['accord_key'][['age', 'female', 'white']]
demoVars['white'] = 1*(demoVars['white'] == 'White')

# blood pressure and heart rate
select = subSet(dataDict['bloodpressure'])
bpVars = dataDict['bloodpressure'][select].iloc[:,1:]

# medications
select = subSet(dataDict['concomitantmeds'])
medVars = dataDict['concomitantmeds'][select].iloc[:,1:]
select = np.sum(medVars, axis = 0)/len(medVars) > 0.05
medVars = medVars.loc[:,select]

# events and times (only primary outcome and all-mortality for the moment)
dataDict['cvdoutcomes'].rename({'censor_po': 'event_primary', 'fuyrs_po': 't_primary',
                                'censor_tm': 'event_death', 'fuyrs_tm': 't_death'}, axis = 1, inplace = True)
events = 1 - dataDict['cvdoutcomes'][['event_primary', 'event_death']]
times = dataDict['cvdoutcomes'][['t_primary', 't_death']]

# hemoglobin
select = subSet(dataDict['bba1c'])
hemoVars = dataDict['bba1c'][select].iloc[:,[1]]

# lipids
select = subSet(dataDict['lipids'])
lipidVars = dataDict['lipids'][select].iloc[:,1:]

# other labs
select = subSet(dataDict['otherlabs'])
labVars = dataDict['otherlabs'][select].iloc[:,1:]


#########################################################################################################################

# combine into single dataframe
data = pd.concat([demoVars.index.to_frame(), demoVars, hemoVars, lipidVars, labVars, bpVars, medVars, actions, events, times], axis = 1)

# create Data object and pickle
info = {'x': list(demoVars.columns) + list(hemoVars.columns) + list(lipidVars.columns) + list(bpVars.columns) + list(medVars.columns),
        'id': ['MaskID'],
        'y': list(events.columns),
        'a': list(actions.columns),
        't': list(times.columns)}
split = [0.7, 0, 0.3]
data = Data(data, info, split)
save(data, '../accord_day0')

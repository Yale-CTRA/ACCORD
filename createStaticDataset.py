"""
This is a simple script to preprocess the data into a static, one-row-per-patient format
This file should be run before any anaylysis is performed since it creates the dataset
Created by: Aditya Biswas
"""

import os
import sys
root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
#root = os.path.join(os.path.expanduser('~'), 'Projects')       for use when in the interpreter
sys.path.append(root)
dataFolder = os.path.join(root, 'ACCORD', 'Data')
os.chdir(os.path.join(dataFolder, 'Original Data'))

from Helper.containers import Data as DataContainer
from Helper.utilities import save

import numpy as np
import pandas as pd

## load every ACCORD file into dictionary with lowercase name as key (converted sas -> csv beforehand)
dataDict = {}
contents = os.listdir()
for file in contents:
    name, ext =  os.path.splitext(file)
    if ext == '.csv':
        dataDict[name.lower()] = pd.read_csv(file, index_col = 0)

###################################################################################################
subSet = lambda data: data['Visit'] == 'BLR'  # returns boolean index for all first patient records

## treatment markers
m = len(dataDict['accord_key'])
actions = np.zeros((m, 3))      # three treatment trials in one study
colIndex = dataDict['accord_key'].columns.get_loc('treatment')
for i in range(m):
    # strings in the format of 'Intensive Glycemia/Lipid Control'
    # each person enrolled in only one 2/3 trials
    # those not assigned to a trial get a 0 b/c effectively equivalent to controls
    string = dataDict['accord_key'].iloc[i,colIndex]
    slashLoc = string.index('/')
    firstString, secondstring = string[:slashLoc], string[slashLoc+1:]
    
    # everyone enrolled in this trial
    # two checks required since they mispelled glycemia a bunch of times
    if firstString == 'Intensive Glycemia' or firstString == 'Intensive Gylcemia':
        actions[i,0] = 1
        
    # everyone enrolled in one or the other of these two trials
    if secondstring == 'Intensive BP':
        actions[i,1] = 1
    if secondstring == 'Lipid Fibrate':
        actions[i,2] = 1

actions = pd.DataFrame(actions, index = dataDict['accord_key'].index,
                       columns = ['glycemia', 'bp', 'lipid'])


## demographics
dataDict['accord_key'].rename({'baseline_age': 'age', 'raceclass2': 'white'}, 
                                axis = 1, inplace = True)
demoVars = dataDict['accord_key'][['age', 'female', 'white']]
demoVars['white'] = 1*(demoVars['white'] == 'White')

## blood pressure and heart rate
select = subSet(dataDict['bloodpressure'])
bpVars = dataDict['bloodpressure'][select].iloc[:,1:]

## medications
select = subSet(dataDict['concomitantmeds'])
medVars = dataDict['concomitantmeds'][select].iloc[:,1:]
select = np.sum(medVars, axis = 0)/len(medVars) > 0.05
medVars = medVars.loc[:,select]

## events and times (only primary composite and all-mortality for the moment)
dataDict['cvdoutcomes'].rename({'censor_po': 'event_primary', 'fuyrs_po': 't_primary',
                                'censor_tm': 'event_death', 'fuyrs_tm': 't_death'}, 
                                    axis = 1, inplace = True)
events = 1 - dataDict['cvdoutcomes'][['event_primary', 'event_death']]
times = dataDict['cvdoutcomes'][['t_primary', 't_death']]

## hemoglobin
select = subSet(dataDict['bba1c'])
hemoVars = dataDict['bba1c'][select].iloc[:,[1]]

## lipids
select = subSet(dataDict['lipids'])
lipidVars = dataDict['lipids'][select].iloc[:,1:]

## other labs
select = subSet(dataDict['otherlabs'])
labVars = dataDict['otherlabs'][select].iloc[:,1:]


##############################################################################################

## combine into single dataframe
# leaving out medVars for the moement
data = pd.concat([demoVars.index.to_frame(), demoVars, hemoVars, lipidVars, labVars,
                  bpVars, actions, events, times], axis = 1)

## create Data object and pickle
info = {'x': list(demoVars.columns) + list(hemoVars.columns) + list(lipidVars.columns) + \
                list(bpVars.columns) + list(actions.columns)[1:], #+ list(medVars.columns),
        'id': 'MaskID',
        'y': list(events.columns),
        'a': list(actions.columns)[0],
        't': list(times.columns)}
split = [0.7, 0, 0.3]       # 70% train, 30% test
data = DataContainer(data, info, split)
name = 'accord_day0'
save(data, loc = dataFolder, name = name)
print('\nData object pickled to: ', os.path.join(dataFolder, name))

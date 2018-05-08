# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 12:25:26 2018

@author: adityabiswas
"""

def main():
    
    import os
    import sys
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    #root = os.path.join(os.path.expanduser('~'), 'Projects')       for use when in the interpreter
    sys.path.append(root)
    
    import numpy as np
    import pandas as pd
    import time
    
    from ITE_Estimators.Survival.VRtrees import RandomForest as RF
    from Helper.metrics import strategyGraph
    from Helper.utilities import load, getHrsMins
    
    dataFolder = os.path.join(root, 'ACCORD', 'Data')
    data = load(dataFolder, 'accord_day0')
    
    
    IDs = np.sort(np.concatenate([data.train['id'], data.test['id']]))
    allResults = pd.DataFrame(index = IDs)
    
    
    experimentName = 'temp2'
    experimentFolder = os.path.join(dataFolder, 'experiments', experimentName)
    if not os.path.exists(experimentFolder):
        os.makedirs(experimentFolder)
        os.makedirs(os.path.join(experimentFolder, 'graphs'))
    
    totalTime = 0
    iterations = 1
    auucs = np.zeros(iterations)
    varImportances = pd.DataFrame(index = data.info['x'])
    for i in range(iterations):
        data.refresh(seed = i)
        
        Xtrain = data.train['x']
        Atrain = data.train['a'] == 1   # glucose intervention
        Otrain = np.stack([data.train['y'][:,0], data.train['t'][:,0]], axis = 1)
        
        ## fit
        numTrees = 500
        start = time.time()
        model = RF(numTrees, minGroup = 5, alpha = 0.5, verbose = True)
        model.fit(Xtrain, Atrain, Otrain, colNames = data.info['x'])
        
    
        ## evaluate
        dataEval = data.test
        results = model.predict(dataEval['x'])
        avgLeaves = model.getNumLeaves()
        Y, T, A = dataEval['y'][:,0], dataEval['t'][:,0]*365.25, dataEval['a']
        #results = np.mean(results, axis = 1)
        auuc = strategyGraph(results, Y, T, A, tau = 7*365.25, bins = 20, 
                             plot = False, save = os.path.join(experimentFolder, 'graphs', str(i+1) + '.png'))
        
        
        stop = time.time()
        totalTime += (stop-start)/(60*60)
        avgTime = totalTime/(i+1)
        timeLeft = avgTime*(iterations - (i+1))
        print('Iteration: ', i, ' | Time Completed: ', getHrsMins(totalTime),
              ' | Time Left: ', getHrsMins(timeLeft))
        
        evalIDs = dataEval['id']
        results = pd.Series(results, index = evalIDs)
        allResults[i+1] = results
        auucs[i] = auuc
        print('Performance: ', np.round(auuc, decimals = 2), ' | Running avg: ', 
              np.round(np.mean(auucs[:i+1]), decimals = 2), ' | Avg Leaves: ', avgLeaves, '\n')
        
        ## var Importances
        varImportances[i] = model.getVarImportances()
    
    ## after all iterations have run
    auucs = pd.Series(auucs)
    auucs.to_csv(os.path.join(experimentFolder, 'auucs.csv'))
    allResults.to_csv(os.path.join(experimentFolder, 'results.csv'))
    varImportances.to_csv(os.path.join(experimentFolder, 'importances.csv'))

if __name__ == "__main__":
    main()
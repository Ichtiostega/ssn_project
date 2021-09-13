# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WLpDCpfDxSqCYVb7s4Fg6oqzyacPCli-
"""
import multiprocessing
from datetime import datetime
from data_provider import DataProvider
from rnn import Rnn
from evaluator import Evaluator
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
import pickle

provider = DataProvider("Foreign_Exchange_Rates.csv")
def reduce_output(scenario):
  def helper_mse(e1, e2):
    if isinstance(e1, dict):
       return e1['mse'] + e2['mse']
    else:
       return e1 + e2['mse']

  def helper_percent(e1, e2):
    if isinstance(e1, dict):
       return e1['percent error'] + e2['percent error']
    else:
       return e1 + e2['percent error']

  result = {}
  result['epochs'] = scenario[0]['epochs']
  result['batch_size'] = scenario[0]['batch_size']
  result['input_days'] = scenario[0]['input_days']
  result['check_days'] = scenario[0]['check_days']
  result['mse'] = reduce(helper_mse, scenario)/len(scenario)
  result['percent error'] = reduce(helper_percent, scenario)/len(scenario)
  return result

def tester(input_days_amounts, check_days_amounts, epochs_values, batch_sizes, seed=123456, data_size=1000):
    tests_data = []
    for input_days in input_days_amounts:
        x_train, y_train, x_valid, y_valid, x_test, y_test = provider.get_data_sets(base_size=input_days, seed=seed, size=data_size)
        for epochs in epochs_values:
            for batch_size in batch_sizes:
                model = Rnn(input_days, epochs, batch_size)
                model.compile()
                model.train(x_train, y_train, x_valid, y_valid)
                evaluator = Evaluator(model)
                for check_days in check_days_amounts:
                    print(f'Calculating for epochs {epochs}, batch size {batch_size}, input days {input_days}, check_days {check_days}...')
                    test_info = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'input_days': input_days,
                        'check_days': check_days,
                    }
                    _, _, _, _, x_test, y_test = provider.get_data_sets(base_size=input_days, seed=seed, result_size=check_days)
                    if check_days == 1:
                        test_info['mse'], test_info['percent error'] = evaluator.evaluate_simple(x_test, y_test)
                    else:
                        test_info['mse'], test_info['percent error'] = evaluator.evaluate_long_term(x_test, y_test)
                    tests_data.append(test_info)
                    print(f'Done calculating for epochs {epochs}, batch size {batch_size}, input days {input_days}, check_days {check_days}...')
    return tests_data
    
def main():
    data = DataProvider("Foreign_Exchange_Rates.csv")
    provider = data
    input_sizes = [3, 4, 5]
    epochs = 100
    batch_size = 100
    seed = 123456
    tester_output = []
    #tester_args = [[[1, 2, 4, 8, 16], [1, 3, 7, 30], [400], [200]]]*10
    tester_args = [[[2], [30], [100], [200]]]*1
    now = datetime.now()
    with multiprocessing.Pool(10) as p:
        tester_output = p.starmap(tester, tester_args)
    elapsed_time = datetime.now() - now
    print(elapsed_time)
    tester_output = list(zip(*tester_output))
    tester_output = list(map(reduce_output, tester_output))

    print(tester_output)
    with open('tester_output.pckl', 'wb') as f:
        pickle.dump(tester_output, f)
    
if __name__ == "__main__":
    main()

#import time
#from datetime import datetime
#from joblib import Parallel, delayed
#import multiprocessing
#num_cores = multiprocessing.cpu_count()
#def timer(v):
#    time.sleep(2)
#    t = datetime.now()
#    time.sleep(2)
    #print(t, v)
#    return t, v

#args = list(range(10))

# Parallel(n_jobs=num_cores)(delayed(timer)(*args) for _ in range(4))

#with multiprocessing.Pool(10) as p:
#    rslt = p.map(timer, args)

#rslt
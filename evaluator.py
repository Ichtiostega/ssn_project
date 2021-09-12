import numpy as np
class Evaluator:
    def __init__(self, model):
        self.model = model
        
    def evaluate_simple(self, x_test, y_test):
        results = []
        for data, expected_result in zip(x_test, y_test):
            data = np.array([data])
            results.extend((self.model.simple_predict(data)-expected_result)**2)
        mse = sum(results) / len(results)
        return mse

    def evaluate_long_term(self, x_test, y_test):
        results = []
        for data, expected_results in zip(x_test, y_test):
            data = np.array([data])
            results.extend((self.model.predict_few_days(data, len(expected_results))-expected_results)**2)
        mse = sum(results) / len(results)
        return mse
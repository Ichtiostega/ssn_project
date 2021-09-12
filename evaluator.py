import numpy as np
class Evaluator:
    def __init__(self, model):
        self.model = model
        
    def evaluate_simple(self, x_test, y_test):
        results = []
        for data, expected_result in zip(x_test, y_test):
            data = np.array([data])
            model_result = self.model.simple_predict(data)
            model_result = np.array(model_result).flatten()
            results.extend((model_result-expected_result)**2)
        mse = sum(results) / len(results)
        return mse

    def evaluate_long_term(self, x_test, y_test):
        results = []
        for data, expected_results in zip(x_test, y_test):
            data = np.array([data])
            model_result = self.model.predict_few_days(data, len(expected_results))
            model_result = np.array(model_result).flatten()
            results.extend((model_result-expected_results)**2)
        mse = sum(results) / len(results)
        return mse
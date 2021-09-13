import numpy as np
class Evaluator:
    def __init__(self, model):
        self.model = model
        
    def evaluate_simple(self, x_test, y_test):
        results_mse = []
        results_percent = []
        for data, expected_result in zip(x_test, y_test):
            data = np.array([data])
            model_result = self.model.simple_predict(data)
            model_result = np.array(model_result).flatten()
            results_percent.extend(abs(abs(model_results)-abs(expected_results))/max(model_result, expected_result)*100)
            results_mse.extend((model_result-expected_result)**2)
        mse = sum(results_mse) / len(results_mse)
        percent_error = sum(results_percent) / len(results_percent)
        return mse, percent_error

    def _calculate_percent_error(self, model_results, expected_results):
        result = []
        difference = abs(abs(model_results)-abs(expected_results))
        for index in range(len(model_results)):
            result.append(difference[index]/(max(abs(model_results[index]), abs(expected_results[index]))+0.00000001)*100)
        return result

    def evaluate_long_term(self, x_test, y_test):
        results_mse = []
        results_percent = []
        for data, expected_results in zip(x_test, y_test):
            data = np.array([data])
            model_results = self.model.predict_few_days(data, len(expected_results))
            model_results = np.array(model_results).flatten()
            results_percent.extend(self._calculate_percent_error(model_results, expected_results))
            
            results_mse.extend((model_results-expected_results)**2)
        mse = sum(results_mse) / len(results_mse)
        percent_error = sum(results_percent) / len(results_percent)
        return mse, percent_error
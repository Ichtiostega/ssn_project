from numpy import float64
import pandas as pd

class DataProvider:
    def __init__(self, data_file):
        with open(data_file) as data_file:
            self.data = pd.read_csv(data_file, parse_dates=[1], na_values=['ND']).dropna()
            self.data = self.data[['Time Serie', 'EURO AREA - EURO/US$']]
            self.data = self.data.rename(columns={'Time Serie': 'date', 'EURO AREA - EURO/US$': 'value'})
            self.data['value'] = self.data['value'].astype(float64)

    def get_data_sets(self, base_size=4, proportions=(0.7, 0.2, 0.1)):
        pass

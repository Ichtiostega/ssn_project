import pandas as pd

class DataProvider:
    def __init__(self, data_file):
        with open(data_file) as data_file:
            data = pd.read_csv(data_file, parse_dates=[1], na_values=['ND'])
            self.data = data[['Time Serie', 'EURO AREA - EURO/US$']]
            self.data = self.data.rename(columns={'Time Serie': 'date', 'EURO AREA - EURO/US$': 'value'})

    def get_data_sets(self, proportions=(0.7, 0.2, 0.1)):
        pass


d = DataProvider("Foreign_Exchange_Rates.csv").data
print(d['value'].max())

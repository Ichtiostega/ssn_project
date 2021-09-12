from copy import deepcopy
from math import isnan
import random
from functools import partial

import pandas as pd


class DataProvider:
    @staticmethod
    def _fill(data):
        new_data = deepcopy(data)
        for index, value in data.iteritems():
            if isnan(value):
                next_viable_index = index + 1
                while isnan(data.at[next_viable_index]):
                    next_viable_index += 1
                new_data.at[index] = new_data.at[index - 1] + (
                    new_data.at[next_viable_index] - new_data.at[index - 1]
                ) / (next_viable_index - (index - 1))
            else:
                new_data.at[index] = value
        return new_data

    def __init__(self, data_file):
        with open(data_file) as data_file:
            self.data = pd.read_csv(data_file, parse_dates=[1], na_values=["ND"])
            self.data = self.data[["Time Serie", "EURO AREA - EURO/US$"]]
            self.data = self.data.rename(
                columns={"Time Serie": "date", "EURO AREA - EURO/US$": "value"}
            )
            self.data["value"] = self.data["value"].astype(float)
            self.data["date"] = pd.to_datetime(self.data["date"])
            self.data["value"] = DataProvider._fill(self.data["value"])
            self.normalize_data()

    def get_data_sets(
        self,
        base_size=4,
        result_size=1,
        proportions=(0.7, 0.2, 0.1),
        seed=None,
        size=1000,
    ):
        def _nn_data_chunk_gen(index, base_size, result_size):
            x = []
            y = []
            for i in range(index, index + base_size):
                x.append(self.data["value"].at[i])
            for i in range(index + base_size, index + base_size + result_size):
                y.append(self.data["value"].at[i])
            return x, y

        random.seed(seed)
        get_chunk = partial(
            _nn_data_chunk_gen, base_size=base_size, result_size=result_size
        )
        indexes = random.sample(range(len(self.data) - base_size - 1), k=size)

        train_ids = indexes[: int(size * proportions[0])]
        validate_ids = indexes[
            int(size * proportions[0]) : int(size * proportions[0])
            + int(size * proportions[1])
        ]
        test_ids = indexes[int(size * proportions[0] + int(size * proportions[1])) :]

        train_data = list(map(get_chunk, train_ids))
        validate_data = list(map(get_chunk, validate_ids))
        test_data = list(map(get_chunk, test_ids))

        return train_data, validate_data, test_data

    def normalize_data(self):
        self.data["value"] = self.data["value"] - self.data["value"].min()
        self.data["value"] = self.data["value"] / (self.data["value"].max())

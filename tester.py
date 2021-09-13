from data_provider import DataProvider
from evaluator import Evaluator
from rnn import Rnn


def main():
    provider = DataProvider(data_file="Foreign_Exchange_Rates.csv")
    input_sizes = [3, 4, 5]
    epochs = 100
    batch_size = 100
    seed = 123456
    for input_size in input_sizes:
        x_train, y_train, x_valid, y_valid, x_test, y_test = provider.get_data_sets(
            base_size=input_size, seed=seed
        )

        model = Rnn(input_size, epochs, batch_size)
        model.compile()
        model.train(x_train, y_train, x_valid, y_valid)

        evaluator = Evaluator(model)
        print(f"input size: {input_size}\ndays checked: 1")
        print(evaluator.evaluate_simple(x_test, y_test))
        for days_to_check in [2, 3, 4, 5]:
            _, _, _, _, x_test, y_test = provider.get_data_sets(
                base_size=input_size, seed=seed, result_size=days_to_check
            )
            print(f"input size: {input_size}\ndays checked: {days_to_check}")
            print(evaluator.evaluate_long_term(x_test, y_test))
            print()


if __name__ == "__main__":
    main()

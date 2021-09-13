from matplotlib import pyplot as plt

def plot_prediction_errors(data):
    tmp = {}
    for test in data:
        val = tmp.setdefault(test['epochs'], {}).setdefault(test['batch_size'], {}).setdefault(test['input_days'], {}).setdefault(test['check_days'], {})
        val['mse'] = test['mse']
        val['percent error'] = test['percent error']

    for epochs in tmp:
        for batch_size in tmp[epochs]:
            for input_days in tmp[epochs][batch_size]:
                mses = []
                pes = []
                x = []
                for check_days in tmp[epochs][batch_size][input_days]:
                    x.append(check_days)
                    mses.append(tmp[epochs][batch_size][input_days][check_days]['mse'])
                    pes.append(tmp[epochs][batch_size][input_days][check_days]['percent error'])
                
                plt.plot(x, mses)
                plt.title(f"Input days: {input_days}")
                plt.ylabel("mse")
                plt.xlabel("days predicted")
                plt.savefig(f"Graph_mse_{epochs}_{batch_size}_{input_days}.png")
                plt.clf()
                plt.plot(x, pes)
                plt.title(f"Input days: {input_days}")
                plt.ylabel("Percent Error")
                plt.xlabel("days predicted")
                plt.savefig(f"Graph_pe_{epochs}_{batch_size}_{input_days}.png")
                plt.clf()

def plot_prediction_errors_at_once(data):
    tmp = {}
    for test in data:
        val = tmp.setdefault(test['epochs'], {}).setdefault(test['batch_size'], {}).setdefault(test['input_days'], {}).setdefault(test['check_days'], {})
        val['mse'] = test['mse']
        val['percent error'] = test['percent error']

    for epochs in tmp:
        for batch_size in tmp[epochs]:
            for input_days in tmp[epochs][batch_size]:
                mses = []
                pes = []
                x = []
                for check_days in tmp[epochs][batch_size][input_days]:
                    x.append(check_days)
                    mses.append(tmp[epochs][batch_size][input_days][check_days]['mse'])
                    pes.append(tmp[epochs][batch_size][input_days][check_days]['percent error'])
                
                plt.plot(x, mses, label=f'Input days: {input_days}')
    plt.legend()
    plt.ylabel("mse")
    plt.xlabel("days predicted")
    plt.savefig(f"Graph_mse.png")


def plot_comparison_graph(model, base_days, prediction_days, starting_date, file_name=None):
    data = provider.data
    start = data[data['date'] == starting_date].index[0]
    values_from_data = list(data['value'][start:start+base_days+prediction_days])
    values_from_model = list(provider.normalized_data()['value'][start:start+base_days])
    for _ in range(prediction_days):
        input = np.empty((1,base_days,1), float)
        for i, el in enumerate(values_from_model[-base_days:]):
            input[0][i][0] = el
        values_from_model.append(float(model.model(input)[0][0]))
    values_from_model = list(map(lambda x: provider.denormalize(x), values_from_model))
    plt.plot(list(data['date'][start:start+base_days+prediction_days]), values_from_data)
    plt.plot(list(data['date'][start:start+base_days+prediction_days]), values_from_model)
    file_name = file_name if file_name else f"Graph_comparison_{base_days}_{prediction_days}.png"
    plt.savefig(file_name)

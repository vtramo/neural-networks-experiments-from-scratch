from nnkit.plotlib import load_histories_from_file, plot_training_history

if __name__ == '__main__':
    histories = load_histories_from_file('net_32ReLU-train-histories-kfold.pkl')
    for history in histories:
        plot_training_history(history, metrics={'test_accuracy'})

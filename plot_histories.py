from nnkit.plotlib import load_histories_from_file, load_histories_from_files, plot_training_histories, plot_training_history

if __name__ == '__main__':
    files = ['irpropplus_125000_batch_256_10_tanh_softmax_40epochs.pkl', 'rpropminus_125000_batch_256_10_tanh_softmax_40epochs.pkl', 'rpropplus_125000_batch_256_10_tanh_softmax_40epochs.pkl', 'irpropminus_125000_batch_256_10_tanh_softmax_40epochs.pkl','sgdmomentum_125000_online_256_10_tanh_softmax_40epochs.pkl']
    histories = load_histories_from_files(files)
    plot_training_histories(histories, metric='training_cross_entropy_loss', ylabel='Loss', title='Training Loss')

                

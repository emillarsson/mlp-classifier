import time
import random
import os 
import signal

import tensorflow as tf
import numpy as np

from network import NeuralNetwork
from data_manager import DataManager
from stats import Stats

device = '/cpu:0'

flags = tf.app.flags

# Set random seed
flags.DEFINE_integer('random_seed', 1, 'Random seed.')

# Training parameters
flags.DEFINE_integer('folds', 5, 'Number of folds.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 32, 'Size of batches for training.')
flags.DEFINE_boolean('use_additional_data', False, 'Make use of additional, incomplete, data.')

# Network parameters
flags.DEFINE_string('optimizer', 'gradient', 'Choose which optimizer to use. [gradient, adam, rmsprop]')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate of the optimizer.')
flags.DEFINE_float('l2_regulator', 0.01, 'Regulating the L2 loss of the weights.')
flags.DEFINE_float('drop_out', 0.5, 'Set drop out factor in layers.')


settings = flags.FLAGS

# Fully connected layers for network
hiddens = [1000,500,32]



def signal_handler(signal, frame):
    global interrupted    
    print('Finishing epoch and closing...')
    interrupted = True



def main(Argv=None):
    random.seed(settings.random_seed)
    np.random.seed(settings.random_seed)

    global interrupted
    interrupted = False
    signal.signal(signal.SIGINT, signal_handler)

    summary_dir = 'logs/log '+ time.strftime("%H:%M:%S")

    # Sets up a data manager which handles features and labels
    data_manager = DataManager(batch_size=settings.batch_size,
                               n_folds=settings.folds, 
                               use_additional_data=settings.use_additional_data)

    inputs, classes = data_manager.get_training_size()

    test_data = data_manager.load_test_data()

    # Set up and initialize network
    network = NeuralNetwork(device, 'network', inputs, classes, hiddens, settings.optimizer, settings.learning_rate, settings.l2_regulator, settings.drop_out)

    # Set up and initialise tensorboard
    scalar_tags = []
    for fold in range(settings.folds):
        scalar_tags.append('network/fold-{}/loss'.format(fold))
        scalar_tags.append('network/fold-{}/train-accuracy'.format(fold))
        scalar_tags.append('network/fold-{}/test-accuracy'.format(fold))

    summary_writer = tf.summary.FileWriter(summary_dir, network.sess.graph)

    stats = Stats(sess=network.sess, 
                  summary_writer=summary_writer, 
                  scalar_tags=scalar_tags,
                  histogram_tags=None)

    # Run through folds, reinitialising every new fold
    for fold in range(settings.folds):
        network.init_variables()
        step = 0

        for epoch in range(settings.epochs):
            if interrupted:
                break
            while True:
                done, batch_xs, batch_ys = data_manager.get_next_batch(fold)

                acc = network.get_accuracy(batch_xs, batch_ys)

                _, loss = network.train(batch_xs, batch_ys)

                stats.update({'network/fold-{}/loss'.format(fold): loss, 
                          'network/fold-{}/train-accuracy'.format(fold): acc},
                          None, step=step)
                step += 1

                if done: 
                    break

            batch_test_xs, batch_test_ys = data_manager.get_test(fold)

            test_acc = network.get_accuracy(batch_test_xs, batch_test_ys)
           
            stats.update({'network/fold-{}/test-accuracy'.format(fold): test_acc},
                          None, step=epoch)

            print('Test accuracy: {:.2f} Fold: {} Epoch: {}'.format(100*test_acc, fold,  epoch))

        print('Prediction and saving output...', end='\r')
        predictions = np.argmax(network.predict(test_data), axis=1)

        data_manager.write_prediction(predictions, 'predictions_fold_{}.csv'.format(fold), summary_dir)

        if interrupted:
            break

    # Write out flags to info file
    print('Writing info.txt                          ')
    file = open(summary_dir + '/info.txt',"w") 
    for flag, value in settings.__flags.items():
        file.write(flag + ': ' + str(value) + '\n')
    file.write('\nHiddens: [')
    for hidden in hiddens:
        file.write(' ' + str(hidden))
    file.write(' ]')
    file.close()



if __name__ == '__main__':
    tf.app.run()
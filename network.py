import tensorflow as tf
import tensorflow.contrib.layers as layers

class NeuralNetwork(object):
    def __init__(self, 
                 device, 
                 name, 
                 inputs, 
                 classes, 
                 hiddens, 
                 optimizer,
                 learning_rate, 
                 l2_regulator, 
                 drop_out):

        self.sess = tf.Session()

        with tf.device(device):
            with tf.name_scope(name):
                # Setup input and output placeholders
                with tf.name_scope('input'):
                    self.x = tf.placeholder(tf.float32, shape=[None, inputs])

                with tf.name_scope('output'):
                    self.y_ = tf.placeholder(tf.float32, shape=[None, classes])

                # Set up hidden layers using tensorflow.contrib.layers
                self.hidden_layers = []
                with tf.name_scope('hidden'):
                    self.hidden_layers.append(self.x)
                    for i in range(len(hiddens)):
                        self.hidden_layers.append(layers.fully_connected(self.hidden_layers[i], num_outputs=hiddens[i], activation_fn=tf.nn.relu))
                    self.y = layers.fully_connected(self.hidden_layers[len(hiddens)], num_outputs=classes, activation_fn=tf.nn.relu)


            with tf.name_scope('optimizer'):
                with tf.name_scope('loss'):
                    # Add L2 loss to weights, skip biases
                    vars = tf.trainable_variables()
                    loss_l2 = tf.add_n([tf.nn.l2_loss(var) for var in vars if 'bias' not in var.name]) * l2_regulator

                    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_) + loss_l2)

                    if optimizer.lower() == 'adam':
                        # Adam Optimizer
                        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
                    elif optimizer.lower() == 'rmsprop':
                        # RMSProp
                        self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
                    else: 
                        # Gradient Descent
                        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)


            # Get accuracy with one hot labeled vectors
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize variables
    def init_variables(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # Method for training model
    def train(self, x_input, target_output):
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.x: x_input, self.y_: target_output})
        return _, loss

    # Method for predicting output
    def predict(self, x_input):
        return self.sess.run(self.y, feed_dict={self.x: x_input})
        
    # Method for getting accuracy
    def get_accuracy(self, x_input, target_output):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_input, self.y_: target_output})
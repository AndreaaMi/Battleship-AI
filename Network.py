import tensorflow as tf
import numpy as np

class Network():
    def __init__(self, board_width, board_height, num_ships, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.num_ships = num_ships
        self.num_input_dimension = self.num_ships + 1
        self.board_size = self.board_width * self.board_height
        self.hidden_units = self.board_size
        self.output_units = self.board_size
        self.type = tf.float32

        tf.reset_default_graph()
      
        self.input_dimensions = tf.placeholder(
                self.type, shape=[1, self.num_input_dimension * board_height * board_width])
        self.input_dimensions_reshaped = tf.reshape(
                self.input_dimensions, [-1, board_height, board_width, self.num_input_dimension])
        
        # Konvolucioni slojevi
        self.layer1 = tf.layers.conv2d(inputs=self.input_dimensions_reshaped,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", activation=tf.nn.relu)
        self.layer2 = tf.layers.conv2d(inputs=self.layer1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      activation=tf.nn.relu)
        self.layer3 = tf.layers.conv2d(inputs=self.layer2, filters=self.num_input_dimension,
                                            kernel_size=[1, 1], padding="same",
                                            activation=tf.nn.relu)
        self.layer3_reshaped = tf.reshape(
                self.layer3, [-1, board_height * board_width * self.num_input_dimension])

        self.logits = tf.layers.dense(inputs=self.layer3_reshaped, units=self.board_size)
        self.probabilities = tf.nn.softmax(self.logits)


        self.labels = tf.placeholder(tf.int64)
        self.learning_rate = tf.placeholder(self.type, shape=[])
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
                
        # Gubitak
        self.loss = tf.reduce_sum(tf.multiply(self.learning_rate, self.cross_entropy))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(self.loss)

     
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.session.run(init)

       
        self.saver = tf.compat.v1.train.Saver()
        if model_file is not None:
            print('load model', model_file)
            self.restoreModel(model_file)

    def getBoardProbabilities(self, input_dimensions):
        return self.session.run(
                self.probabilities,
                feed_dict={self.input_dimensions: input_dimensions}
                )

    def trainStep(self, input_dimensions, labels, learning_rate):
        entropy, _ = self.session.run(
                [self.cross_entropy, self.optimizer],
                feed_dict={self.input_dimensions: input_dimensions,
                           self.labels: labels,
                           self.learning_rate: learning_rate
                           })
        return entropy

    def saveModel(self, model_path):
        self.saver.save(self.session, model_path)

    def restoreModel(self, model_path):
        self.saver.restore(self.session, model_path)

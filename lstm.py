import tensorflow as tf
import tensorflow.contrib.rnn as rnn

class lstm(object):
    lstm_hidden_size = 100
    Da = 50
    ip_dimension = 300
    def __init__(
            self, sequence_length, num_classes,
            embedding_size, filter_sizes, num_filters, batch_size=64, l2_reg_lambda=0.0):

        self.sequence_length = sequence_length
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, self.ip_dimension], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # self.x_left = tf.unstack(self.input_x, 14, 1)
        with tf.name_scope("context_vector"):
            self.outputs, self.fw_state = self.getLstmStates(self.input_x)
            self.output = tf.transpose(self.outputs, [1, 0, 2])
            self.last = tf.gather(self.output, int(self.output.get_shape()[0]) - 1)
            # self.Vc = tf.concat(fw_state)
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[self.lstm_hidden_size, num_classes], name="W"))
            # "W",
            # shape=[2*embedding_size, num_classes],
            # initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.last, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(self.losses, name= 'loss')
            # self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def getLstmStates(self, ip):
        with tf.name_scope("lstm"):
            # fw_cell = rnn.LayerNormBasicLSTMCell(num_units = self.lstm_hidden_size,
            #                                      # dropout_keep_prob=0.5
            #                                      )
            fw_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.lstm_hidden_size)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell= fw_cell, input_keep_prob= 0.5)

            out, state = tf.nn.dynamic_rnn(cell = fw_cell,inputs = ip, dtype= tf.float32)

        return out, state



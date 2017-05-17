import tensorflow as tf
import numpy as np
from lstm import lstm
from tensorflow.contrib import learn
import DataExtractor1 as DataExtractor
import pickle
from var import VariationalAutoencoder

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("type_embedding_size", 300, "Dimensionality of type pair embedding (default:128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_sizes2", "1,2,3", "Comma-separated filter sizes (default: '1,2,3')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

run_path = "runs/1495015625"
Cv_filepath = "resources/freepal_relations_with_sentences_no_filtering_test"
TrainDatapath = "resources/freepal_relations_with_sentences_no_filtering_train"
TestDataPath = "resources/freepal_relations_with_sentences_no_filtering_test"
out_domain = "resources/outDomain_Manish.txt"

x_text_train, y_train, max_document_length = DataExtractor.load_data_and_labels_new_encoder(TrainDatapath, True)
x_text_cv, y_dev, _ = DataExtractor.load_data_and_labels_new_encoder(Cv_filepath, False)
x_od, y_od , _ = DataExtractor.load_data_and_labels_glove_custom(out_domain)
x_train = x_text_train
x_dev = x_text_cv
print("Loading data...")
n_samples = len(y_train)
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train : {:d}".format(n_samples))

session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
sess = tf.Session(config=session_conf)

# sess = tf.Session()
cnn = lstm(
            sequence_length=x_train.shape[1],
            # type_sequence_length = entityType_train_Arry.shape[1],
            num_classes=y_train.shape[1],
            # vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

global_step = tf.Variable(0, name="global_step", trainable=False)

def getLSTM_Emb(sess,x_batch, y_batch, writer=None):
    feed_dict1 = {
        cnn.input_x: x_batch,
        cnn.input_y : y_batch,
        cnn.dropout_keep_prob: 1.0
    }
    pred = sess.run(
        [cnn.last],
        feed_dict1)

    return pred


saver = tf.train.Saver()
saver.restore(sess, run_path+"/checkpoints/model-6000")

# lstm_emb = getLSTM_Emb(sess, x_train, y_train)
# lstm_emb_test = getLSTM_Emb(sess, x_dev, y_dev)
# lstm_emb_od = getLSTM_Emb(sess, x_od, y_od)

#### TRAIN AUTOENCODER ####

# max_accuracy = -1
batch_size = FLAGS.batch_size
learning_rate = 0.001
training_epochs = FLAGS.num_epochs

def train(network_architecture, learning_rate=learning_rate,
          batch_size=batch_size):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    return vae

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input= lstm.lstm_hidden_size, #placeholder
         n_z=2)  # dimensionality of latent space


def get_count(cost, threshold):
    cost = np.reshape(cost, newshape= [-1])
    # print cost
    # total = len(cost)
    c_ = cost[np.where(cost<threshold)]
    # print c_
    count = len(c_)
    # print count
    return count

f = open("Result_Log","w")
# Generate batches

with tf.Session() as sess:
    vae_2d = train(network_architecture,learning_rate= learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(vae_2d.cost)
    sess.run(tf.global_variables_initializer())
    batches = DataExtractor.batch_iter(list(zip(x_train, y_train)), batch_size=batch_size, num_epochs=training_epochs)

    for i, batch in enumerate(batches):

        if i ==10:
            print "debug"
        # x_batch, y_batch = zip(*batch)
        x_, y_ = zip(*batch)
        x_batch = getLSTM_Emb(sess, x_, y_)
        x_batch = np.reshape(x_batch, newshape=[-1, lstm.lstm_hidden_size])

        batch_size = len(x_batch)
        avg_cost = 0.
        # total_batch = int(n_samples / batch_size)
        # # Loop over all batches
        # for i in range(total_batch):
        #     batch_xs, _ = mnist.train.next_batch(batch_size)

        # Fit training using batch data
        _, cost = sess.run([opt, vae_2d.cost], feed_dict= {vae_2d.x : x_batch, vae_2d.batch_size : batch_size})
        # Compute average loss
        # max_ = np.max(cost)
        # if max_ > max_loss:
        #     min_loss = max_
        avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        # if batch % 250 == 0:
        # print("Epoch:", '%04d' % (i+1),
        #       "cost=", "{:.9f}".format(avg_cost))
        print("Epoch:" + str(i + 1)+
              "cost="+ str(avg_cost))
    print "######## TEST TRAIN#######"
    print "Average Cost : "+str(avg_cost)
    f.write("Average Cost : "+str(avg_cost))


    # 1495015625
    # min: 2.77708e-10, max: 6.27264e-05, mean: 6.46876826623e-08, sd: 1.47545954798e-06
    # min: 2.80128e-10, max: 9.36649e-05, mean: 1.20761937064e-07, sd: 2.98010369656e-06
    # min: 1.20383e-08, max: 0.0239489, mean: 0.000998617146896, sd: 0.00264052748338

    mean = 6.46876826623e-08
    sd = 1.47545954798e-06
    threshold = mean+sd
    # Applying encode and decode over test set

    in_count = 0
    test_data_x = x_train
    test_data_y = y_train

    # test_data = x_dev
    # max_threshold = 5 * avg_cost
    batches = DataExtractor.batch_iter(list(zip(test_data_x, test_data_y)), batch_size=batch_size, num_epochs=1)
    for i, batch in enumerate(batches):
        # test_x, test_y = zip(*batch)
        x_, y_ = zip(*batch)
        x_batch = getLSTM_Emb(sess, x_, y_)
        x_batch = np.reshape(x_batch, newshape=[-1, lstm.lstm_hidden_size])

        batch_size = len(x_batch)
        # test = np.reshape(test_x, newshape=[1, lstm.lstm_hidden_size])
        cost_test = sess.run(vae_2d.cost_, feed_dict={ vae_2d.x: x_batch, vae_2d.batch_size: 1})
        # print "c1: "+str(i)+" : "+str(cost_test)
        # max_ = np.max(cost_test)
        # if max_ >max_error:
        #     max_error = max_
        f.write("c1: "+str(i)+" : "+str(cost_test)+"\n")
        in_count+= get_count(cost_test, threshold= threshold)
    train_in_count = in_count
    print("IN-DOMAIN : " + str(in_count))
    f.write("IN-DOMAIN : " + str(in_count))
    print ("Total : " + str(len(test_data_x)))
    f.write("Total : " + str(len(test_data_y)))

    print "######## TEST CV#######"
    print "Average Cost : " + str(avg_cost)
    # threshold = max_error
    # Applying encode and decode over test set
    print threshold

    in_count = 0
    test_data_x = x_dev
    test_data_y = y_dev

    # test_data = x_dev
    # max_threshold = 5 * avg_cost
    batches = DataExtractor.batch_iter(list(zip(test_data_x, test_data_y)), batch_size=batch_size, num_epochs=1)
    for i, batch in enumerate(batches):
        # test_x, test_y = zip(*batch)
        x_, y_ = zip(*batch)
        x_batch = getLSTM_Emb(sess, x_, y_)
        x_batch = np.reshape(x_batch, newshape=[-1, lstm.lstm_hidden_size])
        batch_size = len(x_batch)
        # test = np.reshape(test_x, newshape=[1, lstm.lstm_hidden_size])
        cost_test = sess.run(vae_2d.cost_, feed_dict={ vae_2d.x: x_batch, vae_2d.batch_size: 1})
        # print "c2: "+str(i)+" : "+str(cost_test)
        f.write("c2: "+str(i)+" : "+str(cost_test)+"\n")
        in_count+= get_count(cost_test, threshold= threshold)
        #
        # if cost_test < threshold:
        #     in_count += 1
    test_in_count = in_count
    print("IN-DOMAIN : " + str(in_count))
    f.write("IN-DOMAIN : " + str(in_count))
    print ("Total : " + str(len(test_data_x)))
    f.write("Total : " + str(len(test_data_y)))


    print "######## OD  #######"
    print "Average Cost : " + str(avg_cost)
    # threshold = avg_cost
    # Applying encode and decode over test set
    print threshold

    in_count = 0
    test_data_x = x_od
    test_data_y = y_od

    # test_data = x_dev
    # max_threshold = 5 * avg_cost
    batches = DataExtractor.batch_iter(list(zip(test_data_x, test_data_y)), batch_size=batch_size, num_epochs=1)
    for i, batch in enumerate(batches):
        # test_x, test_y = zip(*batch)
        x_, y_ = zip(*batch)
        x_batch = getLSTM_Emb(sess, x_, y_)
        x_batch = np.reshape(x_batch, newshape=[-1, lstm.lstm_hidden_size])

        batch_size = len(x_batch)
        # test = np.reshape(test_x, newshape=[1, lstm.lstm_hidden_size])
        cost_test = sess.run(vae_2d.cost_, feed_dict={ vae_2d.x: x_batch, vae_2d.batch_size: 1})
        # print "c3: "+str(i)+" : "+str(cost_test)
        f.write("c3: "+str(i)+" : "+str(cost_test)+"\n")
        in_count+= get_count(cost_test, threshold= threshold)

        # if cost_test < threshold:
        #     in_count += 1
    od_in_count = in_count
    print("IN-DOMAIN : " + str(in_count))
    f.write("IN-DOMAIN : " + str(in_count))
    print ("Total : " + str(len(test_data_x)))
    f.write("Total : " + str(len(test_data_y)))
    # # Compare original images with their reconstructions
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(examples_to_show):
    #     a[0][i].imshow(np.reshape(test_x[i], (1, x_train[1])))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (1, x_train[1])))
    # f.show()
    # plt.draw()
    # plt.waitforbuttonpress()
    print "Train In Domain / Total: "+str(train_in_count) +" / " + str(len(x_train))
    f.write("Train In Domain / Total: "+str(train_in_count) +" / " + str(len(x_train)))
    print "Test In Domain / Total: " + str(test_in_count) +" / " + str(len(x_dev))
    f.write("Test In Domain / Total: " + str(test_in_count) +" / " + str(len(x_dev)))
    print "OD In Domain / Total: " + str(od_in_count) +" / " + str(len(x_od))
    f.write("OD In Domain / Total: " + str(od_in_count) +" / " + str(len(x_od)))


#
#
#
#
#
# test_file = TrainDatapath
# lines = open(test_file).readlines()
# fw = open('results_in_domain_data_test','w')
# indomain_count = 0
# threshold = 0.01
#
# batches = DataExtractor.batch_iter(list(zip(lstm_emb, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

















#
#
#
#
#
#
#
# # test_batches = DataExtractor.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, FLAGS.num_epochs)
# for i,line in enumerate(lines):
#     if line.__contains__(".xml"):
#         continue
#
#     splt = line.strip("\n").split("\t")
#     pattern = splt[0]
#     # act_rel_id = int(splt[1].strip("\n"))
#     # actual_relation = rel_id_Map[act_rel_id]
#     # emb = DataExtractor.
#     # x_test = np.array(list(vocab_processor.fit_transform(list_sents)))
#     # loss_vector = Test_pattern(sess,x_test)
#     #
#     # if loss_vector > threshold:
#     #     indomain_count+=1
#     # else:
#     #     count+=1
#
#     # if predicted_relation == actual_relation:
#     #     match_loss_mean += loss
#     #     count += 1
#     #     if loss < 0.45:
#     #         cor_count += 1
#     # else:
#     #     mis_match_loss_mean += loss
#     #     mis_match_count += 1
#     # domain = "in-domain"
#     # if loss > 0.45:
#     #     count += 1
#     #     domain = "out-domain"
#     # else:
#     #     indomain_count += 1
#     # fw.write(pattern + "\t" + domain + "\t" + str(loss) + "\t" + str(score[0][prediction[0]])+"\n")
#     # result.append(predicted_relation)
# # print "Out domain count:" + str(count) ##+ "\t" + str(cor_count)
# # print "in domain count" + str(indomain_count)
# # print "match loss mean : " + str(match_loss_mean/count)
# # print "mis match loss mean " + str(mis_match_loss_mean/mis_match_count)
# fw.close()
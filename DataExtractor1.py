import numpy as np
import re
import itertools
from collections import Counter

NERTagsList = []
NERTagsList.append("ORGANIZATION")
NERTagsList.append("LOCATION")
NERTagsList.append("PERSON")
# NERTagsList.append("")
NERTagsList.append("MISC")
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_new(filepath):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    no_freebase_relations = 51
    train_examples = list(open(filepath, "r").readlines())
    train_examples = [s.strip() for s in train_examples]

    labels = np.zeros([len(train_examples),no_freebase_relations])
    x_text = []
    for i,l in enumerate(train_examples):
        splt = l.split("\t")
        pattern = splt[0]
        relation_id = int(splt[1])
        # pattern = pattern + " " + bigramString(pattern)
        # pattern = bigramString(pattern)
        # print i
        relation = splt[2]
        x_text.append(pattern)
        label = np.zeros([no_freebase_relations])
        label[int(relation_id)] = 1
        labels[i,:] = label


    # Split by words

    return [x_text, labels]


def get_max_sequence_length(filename):
    max_sequence_length = 0
    f = file(filename, 'r')
    lines = f.readlines()
    for l in lines:
        sequence_length = 0
        words = l.strip("\n").split(" ")
        for i in enumerate(words):
            sequence_length += 1
        if (sequence_length > max_sequence_length):
            max_sequence_length = sequence_length
    return max_sequence_length

from GloveModel import Glove_Model

# trainPath = "resources/train_web_freepal"
# max_seq = get_max_sequence_length(trainPath)
debug = False
g = Glove_Model(skip= False)
g.model['#crd#'] = np.array(np.random.random(size=g.DIMENSION))
# g.model['#@#'] = np.array(np.random.random(size=g.DIMENSION))
# delimiter = "#@#"
def separateNerTagsFromWindows(window):
    """" This function separates the tags associated with the sentence"""
    "sample sentence : 'throughout the 1990s and under " \
                "#^ORGANIZATION^Croatian #^ORGANIZATION^Democratic #^ORGANIZATION^Union leadership , " \
                "#^LOCATION^Croatia became independent -LRB- 1991 '"
    words = window.split(" ")
    final_words = []
    tag_indx = -10
    for i,word in enumerate(words):
        word_proc = False
        for nerTag in NERTagsList:
            if nerTag in word:
                word_proc = True
                if tag_indx != i-1 :
                    tag_found = nerTag
                    tag_indx = i
                    final_words.append(tag_found.lower())
                    final_words.append(word.replace("#^"+nerTag+"^",""))
                else :
                    tag_indx = i
                    final_words.append(word.replace("#^"+nerTag+"^",""))

        if word_proc == False:
            final_words.append(word)
    return " ".join(final_words)

def load_data_and_labels_new_encoder(filepath, isTrain = True):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    no_freebase_relations = 45
    # embedding_size = 300
    # type_size = 113
    f = open(filepath, "r")

    if debug == True:
        if isTrain == True: num_lines = 5000
        else: num_lines = 500
        train_examples = list(itertools.islice(f, num_lines))
    else:
        train_examples = list(f.readlines())
    train_examples = [s.strip() for s in train_examples]
    train_size = len(train_examples)

    j = 0

    # lw_max_sequence_len = 0
    dp_max_sequence_len = 0
    # rw_max_sequence_len = 0
    # max_type_seq_len = 0
    # total_max =0

    if isTrain:
        for line in train_examples:
            splt = line.split("\t")

            if splt[0] == "1":
                j += 1
                # leftWindow = separateNerTagsFromWindows(splt[1])
                # sequence_len = len(leftWindow.split(" "))
                # if lw_max_sequence_len < sequence_len:
                #     lw_max_sequence_len = sequence_len

                sequence_len = len(splt[2].split(" "))
                if dp_max_sequence_len < sequence_len:
                    dp_max_sequence_len = sequence_len
                #
                # rightWindow = separateNerTagsFromWindows(splt[3])
                # sequence_len = len(rightWindow.split(" "))
                # if rw_max_sequence_len < sequence_len:
                #     rw_max_sequence_len = sequence_len
                # temp_max = lw_max_sequence_len + dp_max_sequence_len + rw_max_sequence_len + 2
                # if temp_max > total_max:
                #     total_max = temp_max
                #
                # type_seq_len = max(len(splt[5].split("####")),len(splt[7].split("####")))
                # if max_type_seq_len < type_seq_len:
                #     max_type_seq_len = type_seq_len

        # if (max_sequence_list != None):
        #     lw_max_sequence_len = max_sequence_list[0]
        #     dp_max_sequence_len = max_sequence_list[1]
        #     rw_max_sequence_len = max_sequence_list[2]
        #     # max_type_seq_len = max_sequence_list[3]

        # max_seq = lw_max_sequence_len + rw_max_sequence_len + dp_max_sequence_len +2
        # train_data_size = j
        # g.max_sequence_length = 1
        # d, d_ = g.get_patternList_stackedEmbedding(delimiter)

        # g.max_sequence_length = max(lw_max_sequence_len, dp_max_sequence_len, rw_max_sequence_len)+1
        g.max_sequence_length = dp_max_sequence_len
    max_seq = g.max_sequence_length
    labels = np.zeros([train_size,no_freebase_relations])
    x_train = np.zeros(shape= [train_size, max_seq, g.DIMENSION])

    for i,l in enumerate(train_examples):
        splt = l.split("\t")
        if splt[0] == "1":
            # temp = []

            # leftWindow = separateNerTagsFromWindows(splt[1])
            dependencyPattern = splt[2]
            # rightWindow = separateNerTagsFromWindows(splt[3])
            relation_id_string = splt[8]

            # g.max_sequence_length = lw_max_sequence_len
            # lw, lengthl = g.get_patternList_stackedEmbedding(leftWindow)
            # lw[max_seq - 1, :] = d
            # temp.append(lw)

            # g.max_sequence_length = dp_max_sequence_len
            dp = g.get_patternList_stackedEmbedding(dependencyPattern)
            # dp[max_seq-1, :] = d
            # temp.append(dp)

            # g.max_sequence_length = rw_max_sequence_len
            # rw, lengthr = g.get_patternList_stackedEmbedding(rightWindow)
            # temp.append(rw)

            # emb = np.array(temp)
            # emb = np.reshape(emb, newshape= [-1, g.DIMENSION])
            emb = dp
            if emb is not None:
                x_train[i] = emb
                # label[int(relation_id)] = 1
                for relation_id in relation_id_string.split(" "):
                    if len(relation_id) > 0:
                        rel_id = int(relation_id)
                        labels[i, rel_id] = 1
                i += 1

        # Split by words

    return x_train, labels, max_seq

# def get_max_sequence_length(filename):
#     max_sequence_length = 0
#     f = file(filename, 'r')
#     lines = f.readlines()
#     for l in lines:
#         sequence_length = 0
#         words = l.strip("\n").split(" ")
#         for i in enumerate(words):
#             sequence_length += 1
#         if (sequence_length > max_sequence_length):
#             max_sequence_length = sequence_length
#     return max_sequence_length

def load_data_and_labels_glove_custom(filepath):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # max_seq = get_max_sequence_length(filepath)
    # g.max_sequence_length = max_seq
    max_seq = g.max_sequence_length
    print "############################################OD max-seq : "+str(max_seq)
    no_freebase_relations = 45
    train_examples = list(open(filepath, "r").readlines())
    train_examples = [s.strip() for s in train_examples]
    train_size = len(train_examples)
    labels = np.zeros([len(train_examples),no_freebase_relations])
    # x_text = []

    x_train = np.zeros(shape= [train_size, max_seq, g.DIMENSION])
    i =0
    for l in enumerate(train_examples):
        if i == 257:
            print "debug"
        splt = l[1].split("\t")
        pattern = splt[1]
        pattern_length = len(pattern.split(" "))
        if pattern_length> max_seq:
            continue
        # relation_id = int(splt[1])
        # pattern = pattern + " " + bigramString(pattern)
        # pattern = bigramString(pattern)
        # print i
        # relation = splt[2]
        # x_text.append(pattern)

        emb = g.get_patternList_stackedEmbedding(pattern)
        if emb is not None:
            x_train[i] = emb
            label = np.zeros([no_freebase_relations])
            # label[int(relation_id)] = 1
            labels[i,:] = label
            i+=1
    # Split by words

    return x_train, labels, max_seq

def load_data_and_labels_new_entity_Type(filepath):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    no_freebase_relations = 113
    max_entityTypes = 42
    train_examples = list(open(filepath, "r").readlines())
    train_examples = [s.strip() for s in train_examples]

    entityTypeArry = np.zeros([len(train_examples), max_entityTypes])
    labels = np.zeros([len(train_examples),no_freebase_relations])
    x_text = []
    for i,l in enumerate(train_examples):
        splt = l.split("\t")
        pattern = splt[0]

        relation_id = splt[1]
        x_text.append(pattern)
        label = np.zeros([no_freebase_relations])
        label[int(relation_id)] = 1
        labels[i,:] = label
        type_sequence_ids = splt[2].split(" ")
        for j,type_id in enumerate(type_sequence_ids):
            # print l
            entityTypeArry[i,j] = int(type_id)


    # Split by words

    return [x_text, labels,entityTypeArry]

def bigramString(strng):
    bigram_String = ""
    words = strng.split(" ")
    for i in range(len(words)-1):
        bigram_String = bigram_String + " " + words[i] + "_" + words[i+1]
    return bigram_String

def loadTraindata(self, filename):
    f = file(filename, 'r')
    lines = f.readlines()
    fb_id = 0
    # self.observation_matrix = np.zeros([self.no_of_relations,self.no_of_ep])
    # self.freebase_observation_matrix= np.zeros([self.no_of_freebase,self.no_of_ep])
    pattern_id = 0
    ep_id = 0
    # self.emb_matrix = np.zeros([self.no_of_relations , word_embedding_size])
    isFB = False
    ep_rel = {}
    x_train = []
    y_train = []
    pattern_count = 0
    no_of_relations = 5
    for j, l in enumerate(lines):
        if l.startswith('REL$/') == True:
            # isFB = True
            split = l.split("\t")
            pattern = split[0]
            entity1 = split[1]
            entity2 = split[2]
            enPair = entity1 + "\t" + entity2
            ep_rel[enPair] = pattern
        else:
            pattern_count += 1

    for j, l in enumerate(lines):
        if l.startswith('REL$/') == True:
            isFB = True
        split = l.split("\t")
        pattern = split[0]
        entity1 = split[1]
        entity2 = split[2]
        enPair = entity1 + "\t" + entity2
        if isFB == False:
            x_train.append(pattern)
            y_train.append(ep_rel[enPair])


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

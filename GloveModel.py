import numpy as np

class Glove_Model:
    max_sequence_length = None
    model = {}
    modelfilePath = '/home/mvidyasa/Desktop/glove.6B.300d.txt'
    # modelfilePath = 'resources/glove.6B.300d.txt'
    DIMENSION = 300
    trainDataPath = ''
    embedding_size = 300
    def __init__(self, max_seq= None, skip = False):
        if max_seq is not None:
            self.max_sequence_length = max_seq
        if skip == False:
            print "Loading Glove Model"
            f = open(self.modelfilePath, 'r')

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                embedding = [float(val) for val in splitLine[1:]]
                self.model[word] = embedding
            print "Done.", len(self.model), " words loaded!"

    def get_pattern_averageEmbedding(self, pattern):
        words = pattern.strip("\n").split(" ")
        Avgemb = np.zeros(self.DIMENSION)
        for word in words:
            if word == "":
                continue
            elif word == "$ARG1":
                continue
            elif word == "$ARG2":
                continue
            else:
                try:
                    Avgemb = Avgemb + self.model[word]
                except:
                    print 0, word
                    return None
        return Avgemb / len(words)

    def get_patternList_avergaeEmbedding(self, lines):
        inputData = np.zeros((len(lines), 300))
        j = 0
        for i, line in enumerate(lines):
            emb = self.get_pattern_averageEmbedding(line)
            if (emb != None):
                inputData[j, :] = emb
                j += 1
        return np.asarray(inputData)

    def get_patternList_ConcatEmbedding(self, line):
        # inputData = np.zeros((len(lines), self.max_sequence_length*300))
        i = 0
        # for i, line in enumerate(lines):

        words = line.strip("\n").split(" ")
        line_emb = np.zeros(self.max_sequence_length*300)
        for word in words:
            word = word.lower()
            if word == "":
                continue
            elif word == "$arg1":
                continue
            elif word == "$arg2":
                continue
            else:
                try:
                    word_emb = self.model[word]
                    if (word_emb is not None):
                        line_emb[i*300:(i+1) * 300] = word_emb
                        i = i + 1

                except:
                    print 0, word
                    return None
            # if(i > 0):
            #     inputData[j, :] = line_emb
            #     j += 1
        return line_emb

    def get_patternList_stackedEmbedding(self, line):
        # inputData = np.zeros((len(lines), self.max_sequence_length*300))
        i = 0
        # for i, line in enumerate(lines):

        words = line.strip("\n").split(" ")
        line_emb = np.zeros(shape=[self.max_sequence_length, 300])
        for word in words:
            if word == "":
                continue
            elif word == "$ARG1":
                continue
            elif word == "$ARG2":
                continue
            else:
                try:
                    word_emb = self.model[word.lower()]
                    if (word_emb is not None):
                        line_emb[i, :] = word_emb
                        i = i + 1
                    else:
                        return None

                except:
                    # print 0, word.lower()
                    return None
                    # if(i > 0):
                    #     inputData[j, :] = line_emb
                    #     j += 1
        return line_emb


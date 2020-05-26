import copy


class Instance:
    __feature = []  # feature value in each dimension
    __len = 0
    __fitness = 0  # fitness of objective function under those features
    __string = ""

    def __init__(self):
        self.__feature = []
        self.__fitness = 0
        self.__len = 0

    # return feature value in index-th dimension
    def getFeature(self, index):
        return self.__feature[index]

    # return features of all dimensions
    def getFeatures(self):
        return copy.copy(self.__feature)

    # set feature value in index-th dimension
    def setFeature(self, index, v):
        self.__feature[index] = v

    # set features of all dimension
    def setFeatures(self, v):
        self.__feature = v

    def getString(self):
        return self.__string

    def setString(self, s):
        self.__string = s

    # return fitness under those features
    def getFitness(self):
        return self.__fitness

    # set fitness
    def setFitness(self, fit):
        self.__fitness = fit

    def getLen(self):
        return self.__len

    def setLen(self, l):
        self.__len = l

    #
    def Equal(self, ins):
        return self.__fitness == ins.getFitness()

    # copy this instance
    def CopyInstance(self):
        copy_ = Instance()
        features = copy.copy(self.__feature)
        copy_.setFeatures(features)
        copy_.setFitness(self.__fitness)
        copy_.setString(self.__string)
        copy_.setLen(self.__len)
        return copy_

    def CopyFromInstance(self, ins_):
        self.__feature = copy.copy(ins_.getFeatures())
        self.__fitness = ins_.getFitness()
        self.__string = ins_.getString()
        self.__len = ins_.getLen()

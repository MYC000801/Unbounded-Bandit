import math
import random
import numpy as np

# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1


# distr: [float] -> (float)
# Normalize a list of floats to a probability distribution.  Gamma is an
# egalitarianism factor, which tempers the distribution toward being uniform as
# it grows from zero to one.
def distr(weights, gamma=0.0):
    gamma=0.01
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)


# log barrier distribution
# from loss to distribution
def lbdistr(loss, lr, gamma):
    loss = np.array(loss) * lr
    # Lagrange multiplier
    minLm = -min(loss) + 1
    maxLm = minLm + len(loss)
    lm = (minLm + maxLm) / 2
    while True:
        loss_bias = np.array(loss) + lm * np.array([1.0] * len(loss))
        estimatedWeight = [1 / l for l in loss_bias]
        if sum(estimatedWeight) > 1:
            minLm = lm
        else:
            maxLm = lm

        if 1.001 > sum(estimatedWeight) > 0.999:
            break
        lm = (minLm + maxLm) / 2
    loss_bias = np.array(loss) + lm * np.array([1.0] * len(loss))
    estimatedWeight = [1 / l for l in loss_bias]
    eW_mixing = add_explore(estimatedWeight, gamma)
    mixing_amount = (np.array(eW_mixing)-estimatedWeight)*gamma
    return eW_mixing, mixing_amount




def add_explore(distr, exploration_rate):
    if exploration_rate==0:
        return tuple(np.array(distr))
    distr = np.array(distr)
    max_index = np.argmax(distr)
    ma = 0
    for i in range(len(distr)):
        if distr[i]<1/exploration_rate:
            ma+=1/exploration_rate
            distr[i]+=1/exploration_rate
    distr[max_index] -= ma
    return tuple(distr)

def sfdistr(loss, lr, gamma):
    loss = np.array(loss) * lr
    # Lagrange multiplier
    minLm =  -min(loss) + 1
    maxLm = minLm + len(loss)
    lm = (minLm + maxLm) / 2
    while True:
        loss_bias = np.array(loss) + lm * np.array([1.0] * len(loss))
        estimatedWeight = [1 / l for l in loss_bias]
        if np.sum(estimatedWeight) > 1:
            minLm = lm
        else:
            maxLm = lm

        if 1.01 > sum(estimatedWeight) > 0.99:
            break
        lm = (minLm + maxLm) / 2
    loss_bias = np.array(loss) + lm * np.array([1.0] * len(loss))
    estimatedWeight = [1 / l for l in loss_bias]
    return tuple((1.0 - gamma) * w + (gamma / len(loss)) for w in estimatedWeight)

def adgdistr(loss, lr, gamma):
    loss = np.array(loss) * lr
    weights = [ math.exp(-l) for l in loss]
    sumWeights = sum(weights)
    estimatedWeight = np.array(weights)/sumWeights
    return tuple((1.0 - gamma) * w + (gamma / len(loss)) for w in estimatedWeight)



#primal to dual
def bank_p2d(loss):
    loss = np.array(loss)
    # Lagrange multiplier
    minLm =  -min(loss) + 1
    maxLm = minLm + len(loss)
    lm = (minLm + maxLm) / 2
    while True:
        loss_bias = np.array(loss) + lm * np.array([1.0] * len(loss))
        estimatedWeight = [1 / l for l in loss_bias]
        if sum(estimatedWeight) > 1:
            minLm = lm
        else:
            maxLm = lm

        if 1.01 > sum(estimatedWeight) > 0.99:
            break
        lm = (minLm + maxLm) / 2
    loss_bias = np.array(loss) + lm * np.array([1.0] * len(loss))
    estimatedWeight = 1/loss_bias
    return tuple(estimatedWeight)

def bank_d2p(distr):
    return tuple(1/w for w in distr)






def mean(aList):
    theSum = 0
    count = 0

    for x in aList:
        theSum += x
        count += 1

    return 0 if count == 0 else theSum / count

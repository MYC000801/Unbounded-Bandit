from probability import distr, lbdistr, draw, bank_p2d, bank_d2p, sfdistr, adgdistr
import math
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Uniform distribution
def uniform(numActions, reward, gamma=0.0):
    loss = [0.0] * numActions
    # clipping threshold
    t = 0
    while True:
        probabilityDistribution = tuple([1.0 / numActions] * numActions)
        choice = draw(probabilityDistribution)
        theReward = reward(choice, t)
        expected_reward = sum(reward(action, t) * probabilityDistribution[action] for action in range(numActions))
        weights = probabilityDistribution

        estimatedReward = -1.0 * theReward / probabilityDistribution[choice]
        loss[choice] += estimatedReward
        yield choice, theReward, expected_reward, weights, loss

        t = t + 1

# ADAHEDGE in "Adaptation to the Range in $ K $-Armed Bandits"
def adagrad(numActions, reward):
    loss = [0.0] * numActions
    weights = [1.0] * numActions
    adagrad_weight = 0.001
    eta = 10
    t = 0
    while True:
        q = distr(weights, 0)
        probabilityDistribution = distr(weights, min(1 / 2, math.sqrt(5 / 2 * numActions / (t + 1))))
        choice = draw(probabilityDistribution)
        theReward = reward(choice, t)
        expected_reward = sum(reward(action, t) * probabilityDistribution[action] for action in range(numActions))

        estimatedloss = -1.0 * theReward / probabilityDistribution[choice]
        loss[choice] += estimatedloss
        try:
            adagrad_weight += q[choice] * estimatedloss + math.log(
                q[choice] * math.exp(-eta * estimatedloss) + 1 - q[choice]) / eta
        except:
            pass

        eta = math.log(numActions) / adagrad_weight
        loss = np.array(loss) - np.array([min(loss)] * numActions)
        weights = [math.exp(-eta * l) for l in loss]
        yield choice, theReward, expected_reward, probabilityDistribution, loss
        t = t + 1


# General Unbounded MAB
def Gbandits(numActions, reward, gamma=0.0):
    loss = [0.0] * numActions
    lr = 1 / 2
    loss_square = 0
    # clipping threshold
    ct = 1
    lrounds = 1200.0

    exploration_rate = 2 * numActions * numActions
    t = 0
    #lrounds = t
    while True:
        if lr == 0:
            probabilityDistribution, mixing_amount = lbdistr(loss, 0,
                                                             2 * numActions * numActions + math.sqrt(
                                                                 numActions * lrounds))
        else:
            probabilityDistribution, mixing_amount = lbdistr(loss, 1 / lr,
                                                             2 * numActions * numActions + math.sqrt(
                                                                 numActions * lrounds))
        choice = draw(probabilityDistribution)
        theReward = reward(choice, t)
        expected_reward = sum(reward(action, t) * probabilityDistribution[action] for action in range(numActions))
        weights = probabilityDistribution
        clip_theReward = min(2 * ct, theReward)
        ct = max(clip_theReward, ct)
        theReward = clip_theReward

        estimatedReward = -1.0 * theReward / probabilityDistribution[choice]
        loss_square += theReward * theReward
        lr = math.sqrt(loss_square / numActions + ct * ct) / 4
        loss[choice] += estimatedReward
        yield choice, theReward, expected_reward, weights, loss

        t = t + 1

# General Unbounded MAB with adaptive exploration rate
def Gbandits_adaptive(numActions, reward, gamma=0.0):
    loss = [0.0] * numActions
    lr = 1 / 2
    loss_square = 0
    # clipping threshold
    ct = 1

    exploration_rate = 0
    t = 0
    while True:

        if lr == 0:
            probabilityDistribution, mixing_amount = lbdistr(loss, 0,
                                                             2 * numActions * numActions + math.sqrt(exploration_rate))
        else:
            probabilityDistribution, mixing_amount = lbdistr(loss, 1 / lr,
                                                             2 * numActions * numActions + math.sqrt(exploration_rate))
        choice = draw(probabilityDistribution)
        theReward = reward(choice, t)
        expected_reward = sum(reward(action, t) * probabilityDistribution[action] for action in range(numActions))
        weights = probabilityDistribution
        clip_theReward = min(2 * ct, theReward)
        ct = max(clip_theReward, ct)
        theReward = clip_theReward

        estimatedReward = -1.0 * theReward / probabilityDistribution[choice]
        exploration_rate += 2 * abs(estimatedReward * mixing_amount[choice])

        loss_square += theReward * theReward
        lr = math.sqrt(loss_square / numActions + ct * ct) / 4
        loss[choice] += estimatedReward
        yield choice, theReward, expected_reward, weights, loss

        t = t + 1


# General Unbounded MAB with best of both rates
def Gbandits_both(numActions, reward, gamma=0.0):
    loss = [0.0] * numActions
    lr = 1 / 2
    loss_square = 0
    # clipping threshold
    ct = 1

    exploration_rate = 0
    t = 0
    while True:

        if lr == 0:
            probabilityDistribution, mixing_amount = lbdistr(loss, 0,
                                                             2 * numActions * numActions + math.sqrt(exploration_rate))
        else:
            probabilityDistribution, mixing_amount = lbdistr(loss, 1 / lr,
                                                             2 * numActions * numActions + math.sqrt(exploration_rate))
        choice = draw(probabilityDistribution)
        theReward = reward(choice, t)
        expected_reward = sum(reward(action, t) * probabilityDistribution[action] for action in range(numActions))
        weights = probabilityDistribution
        clip_theReward = min(2 * ct, theReward)
        ct = max(clip_theReward, ct)
        theReward = clip_theReward

        estimatedReward = -1.0 * theReward / probabilityDistribution[choice]
        exploration_rate += 2 * abs(estimatedReward * mixing_amount[choice])
        if exploration_rate>=1200*numActions:
            exploration_rate = 1200*numActions

        loss_square += theReward * theReward
        lr = math.sqrt(loss_square / numActions + ct * ct) / 4
        loss[choice] += estimatedReward
        yield choice, theReward, expected_reward, weights, loss

        t = t + 1


# SF-MAB in "Scale-free adversarial multi armed bandits"
def SFMAB(numActions, reward, gamma=0.0):
    loss = [0.0] * numActions
    lr = 1 / 2
    eta = numActions
    M = 0
    t = 0
    while True:
        if lr == 0:
            probabilityDistribution = sfdistr(loss, 0, min(1 / 2, math.sqrt(numActions / (t + 1))))
        else:
            probabilityDistribution = sfdistr(loss, eta, min(1 / 2, math.sqrt(numActions / (t + 1))))
        choice = draw(probabilityDistribution)
        theReward = reward(choice, t)
        expected_reward = sum(reward(action, t) * probabilityDistribution[action] for action in range(numActions))
        weights = probabilityDistribution

        estimatedReward = -1.0 * theReward / probabilityDistribution[choice]

        q_update = np.array(bank_d2p(probabilityDistribution))
        q_update[choice] += estimatedReward * eta
        q = np.array(bank_p2d(q_update))
        log_barrier_diff = 0
        for j in range(numActions):
            log_barrier_diff += (
                    math.log(1 / q[j]) - math.log(1 / probabilityDistribution[j]) + 1 / probabilityDistribution[
                j] * (q[j] - probabilityDistribution[j]))
        M += estimatedReward * (probabilityDistribution[choice] - q[choice]) - 1 / eta * log_barrier_diff
        eta = numActions / (1 + M)
        loss[choice] += estimatedReward
        yield choice, theReward, expected_reward, weights, loss

        t = t + 1

# SF-MAB with adaptive exploration rate in "Scale-free adversarial multi armed bandits"
def SFMAB_adaptive(numActions, reward, gamma=0.0):
    loss = [0.0] * numActions
    lr = 1 / 2
    eta = numActions
    M = 0
    t = 0
    gamma = 1 / 2
    Gam = 0
    while True:
        if lr == 0:
            probabilityDistribution = sfdistr(loss, 0, gamma)
        else:
            probabilityDistribution = sfdistr(loss, eta, gamma)
        choice = draw(probabilityDistribution)
        theReward = reward(choice, t)
        expected_reward = sum(reward(action, t) * probabilityDistribution[action] for action in range(numActions))
        weights = probabilityDistribution
        Gam += gamma * abs(theReward) / ((1 - gamma) * probabilityDistribution[choice] + gamma / numActions)
        gamma = numActions / (2 * numActions + Gam)

        estimatedReward = -1.0 * theReward / probabilityDistribution[choice]

        q_update = np.array(bank_d2p(probabilityDistribution))
        q_update[choice] += estimatedReward * eta
        q = np.array(bank_p2d(q_update))
        log_barrier_diff = 0
        for j in range(numActions):
            log_barrier_diff += (
                    math.log(1 / q[j]) - math.log(1 / probabilityDistribution[j]) + 1 / probabilityDistribution[
                j] * (q[j] - probabilityDistribution[j]))
        M += estimatedReward * (probabilityDistribution[choice] - q[choice]) - 1 / eta * log_barrier_diff
        eta = numActions / (1 + M)
        loss[choice] += estimatedReward
        yield choice, theReward, expected_reward, weights, loss

        t = t + 1

# BankerOMD in "Banker Online Mirror Descent: A Universal Approach for Delayed Online Bandit Learning"
def bankerOMD(numActions, reward, gamma=0.0):
    loss = [0.0] * numActions
    ct = 1.0

    t = 0
    d = 0
    ctcounter = 1

    #z = [[0.0] * numActions] * 1258

    zp = [[0.0] * numActions] * 1258
    v = [0.0] * 1258
    # x = [[0.0] * numActions] * 1258
    # b = [[0.0] * numActions] * 1258
    while True:
        #line 4 in BankOMD
        sigma = ctcounter / (math.sqrt(math.log(3 + d / ct / ct) / (3 + d)) * math.sqrt(numActions * math.log(2+t)))

        v[t] = sigma
        b = sigma
        loss_x = [0.0] * numActions
        #loss_x = zp[t-1]

        for s in range(t-1, -1, -1):
            sigma_ts = min(v[s], b)
            b -= sigma_ts
            v[s] -= sigma_ts
            loss_x += zp[s] * sigma_ts / sigma
            if b == 0 or sum(v[:t - 1]) == 0:
                break

        distr = bank_p2d(loss_x)
        #mix_distr = [(1-1/np.sqrt(t+1))*prob+1/np.sqrt(t+1)/numActions for prob in distr]
        #distr = mix_distr
        choice = draw(distr)
        theReward = reward(choice, t)
        d += theReward * theReward
        expected_reward = sum(reward(action, t) * distr[action] for action in range(numActions))
        clip_theReward = min(2 * ct, theReward)
        if theReward>2 * ct:
            ctcounter+=1
        ct = max(clip_theReward, ct)
        theReward = clip_theReward
        estimatedLoss = -1.0 * theReward / distr[choice]
        loss[choice] += estimatedLoss
        z_update = np.array(loss_x)
        z_update[choice] += estimatedLoss / sigma
        zp[t] = z_update
        #z[t] = np.array(bank_p2d(z_update))
        yield choice, theReward, expected_reward, distr, loss

        t = t + 1


def Gbandits_noexp(numActions, reward, gamma=0.0):
    loss = [0.0] * numActions
    lr = 1 / 2
    loss_square = 0
    # clipping threshold
    ct = 1

    exploration_rate = 2 * numActions * numActions
    t = 0
    while True:
        if lr == 0:
            probabilityDistribution, mixing_amount = lbdistr(loss, 0,
                                                             200000)
        else:
            probabilityDistribution, mixing_amount = lbdistr(loss, 1 / lr,
                                                             200000)

        choice = draw(probabilityDistribution)
        theReward = reward(choice, t)
        expected_reward = sum(reward(action, t) * probabilityDistribution[action] for action in range(numActions))
        weights = probabilityDistribution
        clip_theReward = min(2 * ct, theReward)
        ct = max(clip_theReward, ct)
        theReward = clip_theReward

        estimatedReward = -1.0 * theReward / probabilityDistribution[choice]
        loss_square += theReward * theReward
        lr = math.sqrt(loss_square / numActions + ct * ct) / 4
        loss[choice] += estimatedReward
        yield choice, theReward, expected_reward, weights, loss

        t = t + 1
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from data import data_generate_stock, data_generate_amzsales, data_generate_selection
from algorithm import uniform, SFMAB, SFMAB_adaptive, Gbandits, Gbandits_adaptive, bankerOMD, adagrad, Gbandits_both



def regret_comp(index='0'):
    regret_matrix = [[0.0] * 1258] * 8
    std_matrix = [[0.0] * 1258] * 8
    if index == '0':
        rewardVector, sum_reward = data_generate_stock(10)
    elif index == '1':
        rewardVector, sum_reward = data_generate_amzsales(10)
    elif index == '2':
        rewardVector, sum_reward = data_generate_selection(10)

    test_times = 500
    # 500 for fast
    # test_times = 500
    for alg in range(8):
        sum_regret = np.array([0.0] * 1258)
        sum_std = np.array([0.0] * 1258)
        for k in range(test_times):
            random.seed()
            regret = simpleTest(alg, rewardVector)
            sum_regret += np.array(regret) / test_times
            sum_std += np.multiply(np.array(regret), np.array(regret)) / test_times
            print("ALG", alg, ":", k, "round completes")
        regret_matrix[alg] = sum_regret
        std_matrix[alg] = np.sqrt(np.maximum(sum_std - np.multiply(sum_regret, sum_regret),0))

    return np.array(regret_matrix), np.array(std_matrix)


def simpleTest(alg, rewardVector):
    numActions = len(np.array(rewardVector).T)
    numRounds = len(rewardVector)
    rewards = lambda choice, t: rewardVector[t][choice]
    gamma = 0.05

    cumulativeReward = 0
    CumulativeRewards = [0.0] * numActions
    cumulativeExpectReward = 0
    regret_vector = np.array([0.0] * numRounds)
    t = 0
    for (choice, reward, est, weights, loss) in bandits(alg, numActions, rewards, gamma):
        cumulativeReward += reward
        CumulativeRewards += np.array(rewardVector[t])
        cumulativeExpectReward += est

        weakRegret = (max(CumulativeRewards) - cumulativeExpectReward)
        regret_vector[t] = weakRegret

        t += 1
        if t >= numRounds:
            break

    return regret_vector


def bandits(alg, numActions, rewards, gamma):
    if alg == 0:
        return Gbandits(numActions, rewards)
    elif alg == 1:
        return Gbandits_adaptive(numActions, rewards)
    elif alg == 3:
        return adagrad(numActions, rewards)
    elif alg == 4:
        return SFMAB(numActions, rewards)
    elif alg == 5:
        return SFMAB_adaptive(numActions, rewards)
    elif alg == 6:
        return bankerOMD(numActions, rewards)
    elif alg == 7:
        return uniform(numActions, rewards)
    elif alg == 2:
        return Gbandits_both(numActions, rewards)

def test(index):

    labels = ['UMAB-G', 'UMAB-G-A', 'UMAB-G-B', 'AHB', 'SF-MAB', 'SF-MAB-A', 'Banker-OMD', 'Uniform']
    regret_matrix, std_matrix = regret_comp(index)
    # np.save('explo_regret.npy', regret_matrix)
    # np.save('explo_std.npy', std_matrix)

    # regret_matrix = np.load('stock1_regret.npy')
    # std_matrix = np.load('stock1_std.npy')

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.grid(color='k',
             linestyle='--',
             linewidth=1,
             alpha=0.3)

    time = list(range(1258))
    for alg in range(8):
        plt.plot(time, regret_matrix[alg], label=labels[alg], linewidth=2)
        plt.fill_between(time, regret_matrix[alg] - std_matrix[alg], regret_matrix[alg] + std_matrix[alg], alpha=0.1)

    plt.xlim(0, 1258)
    # plt.ylim(8000, 10000)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Number of Rounds', fontsize=24, position="top")
    plt.ylabel('Regret', fontsize=24)
    plt.legend(fontsize=15, frameon=False, loc='upper left')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    #plt.savefig('stock.pdf', dpi=300)
    plt.show()
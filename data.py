import random
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# stocks data
def data_generate_stock(numActions):
    df = pd.read_csv('./all_stocks_5yr.csv')
    # st = ['AAPL', 'ACN', 'ZTS', 'DAL', 'GOOGL', 'AMZN', 'BEN', 'BLL', 'MS', 'VTR']
    st = ['CHK', 'RRC', 'EBAY', 'IBM', 'OXY', 'KMI', 'DISCK', 'SIG', 'AMD', 'FB']
    rewards = [[]] * 1258
    # print(rewards)
    for stock in st:
        stock_data = df[df['Name'].isin([stock])]
        stock_data.reset_index(drop=True, inplace=True)
        reward = [[100 * (stock_data.open[i + 1] - stock_data.open[i]) / stock_data.open[0]] for i in
                  range(len(stock_data.open) - 1)]
        if len(reward) == 1258:
            rewards = [rewards[i] + reward[i] for i in range(len(reward))]
        else:
            print(stock)

    return rewards, np.sum(np.array(rewards), axis=0)


# amazon data
def data_generate_amzsales(numActions):
    # Initilize and normalize data
    df = pd.read_csv('./amazon_sales_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    rewards = [[]] * 10
    st = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for store in st:
        store_data = df[df['Store'].isin([store])]
        grouped = store_data.groupby('Date')
        result = grouped.head(10)
        reward = np.array(result.head(1258)['Weekly_Sales'])
        rewards[store - 1] = reward / 1500.0
    rewards = np.array(rewards).T
    return rewards, np.sum(np.array(rewards), axis=0)


class SGD:
    def __init__(self, lr=0.01, max_iter=1000, batch_size=32, tol=1e-3):
        # learning rate of the SGD Optimizer
        self.learning_rate = lr
        # maximum number of iterations for SGD Optimizer
        self.max_iteration = max_iter
        # mini-batch size of the data
        self.batch_size = batch_size
        # tolerance for convergence for the theta
        self.tolerence_convergence = tol
        # Initialize model parameters to None
        self.theta = None
        self.thetas = [[]] * max_iter

    def fit(self, X, y):
        # store dimension of input vector
        n, d = X.shape
        # Intialize random Theta for every feature
        self.theta = np.random.randn(d)
        for i in range(self.max_iteration):
            self.thetas[i] = np.array(self.theta)
            # Shuffle the data
            indices = np.random.permutation(n)
            X = X[indices]
            y = y[indices]
            # Iterate over mini-batches
            for i in range(0, n, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                grad = self.gradient(X_batch, y_batch)
                self.theta -= self.learning_rate * grad
            # Check for convergence
            if np.linalg.norm(grad) < self.tolerence_convergence:
                break

    # define a gradient functon for calculating gradient
    # of the data
    def gradient(self, X, y):
        n = len(y)
        # predict target value by taking taking
        # taking dot product of dependent and theta value
        y_pred = np.dot(X, self.theta)

        # calculate error between predict and actual value
        error = y_pred - y
        grad = np.dot(X.T, error) / n
        return grad

    def predict(self, X):
        # prdict y value using calculated theta value
        y_pred = np.dot(X, self.theta)
        return y_pred

    def predicts(self, X):
        # prdict y value using calculated theta value
        return [np.dot(X, theta) for theta in self.thetas]

    def loss(self, X, y):
        return [np.linalg.norm(np.dot(X, theta) - y) for theta in self.thetas]


def sgd_loss(lr):
    X = np.random.randn(100, 5)
    y = np.dot(X, np.array([1, 2, 3, 4, 5])) + np.random.randn(100) * 5
    model = SGD(lr=lr, max_iter=1258,
                batch_size=32, tol=1e-3)
    model.fit(X, y)
    return model.loss(X, y)

# meta algorithm selection data
def data_generate_selection(numActions):
    lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 0.0025, 0.025]
    rewards = [[]] * 1258
    # print(rewards)
    for lr in lrs:
        loss = sgd_loss(lr)
        reward = [[-l] for l in loss]
        if len(reward) == 1258:
            rewards = [rewards[i] + reward[i] for i in range(len(reward))]
        else:
            print('error')

    return rewards, np.sum(np.array(rewards), axis=0)

# Non-exploration VS exploration
def data_generate_exp(numActions):
    df = pd.read_csv('./all_stocks_5yr.csv')
    st = ['CHK', 'RRC']
    rewards = [[]] * 1258
    # print(rewards)
    for stock in st:
        stock_data = df[df['Name'].isin([stock])]
        stock_data.reset_index(drop=True, inplace=True)
        if stock == 'CHK':
            reward = [[10 * (i >= 100 and i < 150) + 0.05 * (i >= 150)] for i in
                      range(len(stock_data.open) - 1)]
            # reward = [[1+np.random.normal(0,1)] for i in
            #     range(len(stock_data.open) - 1)]
        else:
            # reward = [[-i % 2+1/2+t.rvs(2.73)] for i in
            #         range(len(stock_data.open) - 1)]
            reward = [[0.5 * (i < 100)] for i in
                      range(len(stock_data.open) - 1)]
            # reward = [[np.random.normal(0,1)] for i in range(len(stock_data.open) - 1)]
        if len(reward) == 1258:
            rewards = [rewards[i] + reward[i] for i in range(len(reward))]
        else:
            print(stock)

    return rewards, np.sum(np.array(rewards), axis=0)


# Non-exploration VS exploration
def data_generate_non_adp(numActions):
    df = pd.read_csv('./all_stocks_5yr.csv')
    st = ['CHK', 'RRC']
    rewards = [[]] * 1258
    # print(rewards)
    for stock in st:
        stock_data = df[df['Name'].isin([stock])]
        stock_data.reset_index(drop=True, inplace=True)
        if stock == 'CHK':
            #reward = [[10 * (i >= 100 and i < 150) + 0.05 * (i >= 150)] for i in
             #         range(len(stock_data.open) - 1)]
            reward = [[1+np.random.normal(0,1)] for i in
                 range(len(stock_data.open) - 1)]
        else:
            # reward = [[-i % 2+1/2+t.rvs(2.73)] for i in
            #         range(len(stock_data.open) - 1)]
            #reward = [[0.5 * (i < 100)] for i in
            #          range(len(stock_data.open) - 1)]
            reward = [[np.random.normal(0,1)] for i in range(len(stock_data.open) - 1)]
        if len(reward) == 1258:
            rewards = [rewards[i] + reward[i] for i in range(len(reward))]
        else:
            print(stock)

    return rewards, np.sum(np.array(rewards), axis=0)

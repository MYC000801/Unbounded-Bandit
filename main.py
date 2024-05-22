import sys

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from data import data_generate_stock, data_generate_amzsales, data_generate_selection
from algorithm import uniform, SFMAB, SFMAB_adaptive, Gbandits, Gbandits_adaptive, bankerOMD, adagrad
import test1,test2, test3

def test(index):
    if index == 'stock':
        test1.test('0')
    elif index == 'amazon':
        test1.test('1')
    elif index == 'selection':
        test1.test('2')
    elif index == 'exploration':
        test2.test()
    elif index == 'adaptive':
        test3.test('0')
    elif index == 'nonadaptive':
        test3.test('1')

if __name__ == "__main__":
    test(sys.argv[1])
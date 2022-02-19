from qulacs.gate import PauliRotation
from qulacs import ParametricQuantumCircuit

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import time 
import datetime
from functools import reduce
import sys
import random

from scipy import optimize
from scipy import special
#import scipy.linalg

from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs import ParametricQuantumCircuit
from qulacs.gate import DenseMatrix
#from qulacs.circuit import QuantumCircuitOptimizer

from qulacs import QuantumState
from qulacs.gate import Identity, X,Y,Z #パウリ演算子
from qulacs.gate import H,S,Sdag, sqrtX,sqrtXdag,sqrtY,sqrtYdag #1量子ビット Clifford演算
from qulacs.gate import T,Tdag #1量子ビット 非Clifford演算
from qulacs.gate import RX,RY,RZ #パウリ演算子についての回転演算
from qulacs.gate import CNOT, CZ, SWAP #2量子ビット演算
from qulacs.gate import U1,U2,U3 #IBM Gate
from qulacs.gate import PauliRotation
from qulacs.gate import merge
from qulacs import Observable
import re

import csv
import copy
import tqdm



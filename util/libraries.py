# Commonly used libraries, not all of which may be useful
import torch,gc
from torchvision import transforms
import scipy
import random
from torch.utils.data import DataLoader,Dataset,ConcatDataset
import pandas as pd
import numpy as np
from scipy.fftpack import rfft,irfft,rfftfreq
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from copy import deepcopy as copy
import csv
from time import process_time
import math
import statsmodels.api as sm
from torch.nn.utils import weight_norm
import sys
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torchvision
from torch.autograd import grad
import math
import time
from sklearn.preprocessing import StandardScaler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
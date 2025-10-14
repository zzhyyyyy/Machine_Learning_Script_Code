import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import ConvNextV2Model, AutoImageProcessor
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import os
import optuna
import copy


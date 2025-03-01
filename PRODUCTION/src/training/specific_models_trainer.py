# src/training/specific_models_trainer.py
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# -*- coding: utf-8 -*-
"""
Harmful Brain Activity Classification (EEG) — Ensemble (EfficientNetB0 + WaveNet)
Harvard Neuro 140. Exported from the original notebook to plain Python for readability.
"""


# ========================================================================
# # Harmful Brain Activity Classification
# #### Student: Breno Freitas
# #### Class: Neuro 240
# ========================================================================

# ========================================================================
# ## Importing libraries
# ========================================================================

import os
import sys
import gc
sys.path.append('/kaggle/input/kaggle-kl-div')
from kaggle_kl_div import score
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0
import pytorch_lightning as pl
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from sklearn.model_selection import KFold, GroupKFold
import torch.nn as nn
import random
import time

from glob import glob
from tqdm import tqdm
from typing import Dict, List

# ========================================================================
# ## Model 1 - EfficientNet
# ========================================================================

# ========================================================================
# ### Data loading
# ========================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

VER = 5
# IF THIS EQUALS NONE, THEN WE TRAIN NEW MODELS
# IF THIS EQUALS DISK PATH, THEN WE LOAD PREVIOUSLY TRAINED MODELS
LOAD_MODELS_FROM = '/kaggle/input/hms-efficientnetb0-pt-ckpts/'
# LOAD_MODELS_FROM = None

USE_KAGGLE_SPECTROGRAMS = True
USE_EEG_SPECTROGRAMS = True

df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = df.columns[-6:]
print('Train shape:', df.shape )
print('Targets', list(TARGETS))
df.head()

df.tail()

# train_temp = df.groupby('eeg_id')[
#     ['spectrogram_id', 'spectrogram_label_offset_seconds']
# ].agg({'spectrogram_id': 'first', 'spectrogram_label_offset_seconds': 'min'})
# train_temp.columns = ['spec_id', 'min']

# train_temp.head()

train = df.groupby('eeg_id')[
    ['spectrogram_id', 'spectrogram_label_offset_seconds']
].agg({'spectrogram_id': 'first', 'spectrogram_label_offset_seconds': 'min'})
train.columns = ['spec_id', 'min']

tmp = df.groupby('eeg_id')[
    ['spectrogram_id','spectrogram_label_offset_seconds']
].agg({'spectrogram_label_offset_seconds' :'max'})
train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape )
train.head()

READ_SPEC_FILES = False

# READ ALL SPECTROGRAMS
PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'
files = os.listdir(PATH)
print(f'There are {len(files)} spectrogram parquets')

if READ_SPEC_FILES:
    spectrograms = {}
    for i,f in enumerate(files):
        if i % 100 == 0:
            print(i, ', ', end='')
        tmp = pd.read_parquet(f'{PATH}{f}')
        name = int(f.split('.')[0])
        spectrograms[name] = tmp.iloc[:,1:].values
else:
    spectrograms = np.load('/kaggle/input/brain-spectrograms/specs.npy',allow_pickle=True).item()

READ_EEG_SPEC_FILES = False

if READ_EEG_SPEC_FILES:
    all_eegs = {}
    for i,e in enumerate(train.eeg_id.values):
        if i % 100 == 0:
            print(i, ', ', end='')
        x = np.load(f'/kaggle/input/brain-eeg-spectrograms/EEG_Spectrograms/{e}.npy')
        all_eegs[e] = x
else:
    all_eegs = np.load('/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()

TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
TARS2 = {x: y for y, x in TARS.items()}


class EEGDataset(Dataset):

    def __init__(self, data, augment=False, mode='train', specs=spectrograms, eeg_specs=all_eegs):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.__getitems__([index])

    def __getitems__(self, indices):
        X, y = self._generate_data(indices)
        if self.augment:
            X = self._augment(X)
        if self.mode == 'train':
            return list(zip(X, y))
        else:
            return X

    def _generate_data(self, indexes):
        X = np.zeros((len(indexes), 128, 256, 8),dtype='float32')
        y = np.zeros((len(indexes), 6),dtype='float32')
        img = np.ones((128, 256),dtype='float32')

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode == 'test':
                r = 0
            else:
                r = int((row['min'] + row['max'])//4)

            for k in range(4):
                # EXTRACT 300 ROWS OF SPECTROGRAM
                img = self.specs[row.spec_id][r:r+300, k*100:(k+1)*100].T

                # LOG TRANSFORM SPECTROGRAM
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)

                # STANDARDIZE PER IMAGE
                ep = 1e-6
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())
                img = (img - m) / (s + ep)
                img = np.nan_to_num(img, nan=0.0)

                # CROP TO 256 TIME STEPS
                X[j, 14:-14, :, k] = img[:, 22:-22] / 2.0

            # EEG SPECTROGRAMS
            img = self.eeg_specs[row.eeg_id]
            X[j, :, :, 4:] = img

            if self.mode != 'test':
                y[j,] = row[TARGETS]

        return X, y

    def _random_transform(self, img):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            # albu.CoarseDropout(max_holes=8,max_height=32,max_width=32,fill_value=0,p=0.5),
        ])
        return composition(image=img)['image']

    def __augment(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self._random_transform(img_batch[i,])
        return img_batch

dataset = EEGDataset(train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

ROWS = 2
COLS = 3
BATCHES = 2

for i, (x, y) in enumerate(dataloader):
    plt.figure(figsize=(20, 8))
    for j in range(ROWS):
        for k in range(COLS):
            plt.subplot(ROWS, COLS, j*COLS + k + 1)
            t = y[j*COLS + k]
            img = torch.flip(x[j*COLS+k, :, :, 0], (0,))
            mn = img.flatten().min()
            mx = img.flatten().max()
            img = (img-mn)/(mx-mn)
            plt.imshow(img)
            tars = f'[{t[0]:0.2f}]'
            for s in t[1:]:
                tars += f', {s:0.2f}'
            eeg = train.eeg_id.values[i*32+j*COLS+k]
            plt.title(f'EEG = {eeg}\nTarget = {tars}',size=12)
            plt.yticks([])
            plt.ylabel('Frequencies (Hz)',size=14)
            plt.xlabel('Time (sec)',size=16)
    plt.show()
    if i == BATCHES-1:
        break

del dataset, dataloader
gc.collect()

# ========================================================================
# ### Training
# ========================================================================

# WEIGHTS_FILE = '/kaggle/input/efficientnet-pytorch/efficientnet_b4_rwightman-23ab8bcd.pth'
WEIGHTS_FILE = '/kaggle/input/efficientnet-pytorch/efficientnet_b0_rwightman-7f5810bc.pth'


class EEGEffnetB0(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.base_model = efficientnet_b0()
        self.base_model.load_state_dict(torch.load(WEIGHTS_FILE))
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, 6, dtype=torch.float32)
        self.prob_out = nn.Softmax()

    def forward(self, x):
        x1 = [x[:, :, :, i:i+1] for i in range(4)]
        x1 = torch.concat(x1, dim=1)
        x2 = [x[:, :, :, i+4:i+5] for i in range(4)]
        x2 = torch.concat(x2, dim=1)

        if USE_KAGGLE_SPECTROGRAMS & USE_EEG_SPECTROGRAMS:
            x = torch.concat([x1, x2], dim=2)
        elif USE_EEG_SPECTROGRAMS:
            x = x2
        else:
            x = x1
        x = torch.concat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)

        out = self.base_model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = F.log_softmax(out, dim=1)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        loss = kl_loss(out, y)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return F.softmax(self(batch), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# import torch
# from numba import cuda

# def free_gpu_cache():
#     torch.cuda.empty_cache()

#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)

# free_gpu_cache()

all_oof = []
all_true = []
valid_loaders = []

gkf = GroupKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):
    print('#'*25)
    print(f'### Fold {i+1}')

    train_ds = EEGDataset(train.iloc[train_index])
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=32, num_workers=3)
    valid_ds = EEGDataset(train.iloc[valid_index], mode='valid')
    valid_loader = DataLoader(valid_ds, shuffle=False, batch_size=64, num_workers=3)

    print(f'### Train size: {len(train_index)}, Valid size: {len(valid_index)}')
    print('#'*25)

    trainer = pl.Trainer(max_epochs=4)
    model = EEGEffnetB0()
    if LOAD_MODELS_FROM is None:
        trainer.fit(model=model, train_dataloaders=train_loader)
        trainer.save_checkpoint(f'EffNet_v{VER}_f{i}.ckpt')

    valid_loaders.append(valid_loader)
    all_true.append(train.iloc[valid_index][TARGETS].values)
    del trainer, model
    gc.collect()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(5):
    print('#'*25)
    print(f'### Validating Fold {i+1}')

    ckpt_file = f'EffNet_v{VER}_f{i}.ckpt' if LOAD_MODELS_FROM is None else f'{LOAD_MODELS_FROM}/EffNet_v{VER}_f{i}.ckpt'
    model = EEGEffnetB0.load_from_checkpoint(ckpt_file)
    model.to(device).eval()
    with torch.inference_mode():
        for val_batch in valid_loaders[i]:
            val_batch = val_batch.to(device)
            oof = torch.softmax(model(val_batch), dim=1).cpu().numpy()
            all_oof.append(oof)
    del model
    gc.collect()

all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)

oof = pd.DataFrame(all_oof.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name='id')
print('CV Score KL-Div for EfficientNetB0 =',cv)

# ========================================================================
# ### Training analysis
# ========================================================================

def calc_conditional_prob(
    preds: np.ndarray,
    gts: np.ndarray,
    weights: np.ndarray | None = None,
    normalize: bool = True,
    norm_axis: int = 0,
    eps=1e-4,
):
    """
    Parameters
    ----------
    preds: (N, C) array of predicted probabilities
    gts: (N, C) array of ground truth probabilities
    weights: (N, ) array of weights

    Returns
    -------
    conditional_matrix: (C, C) array of conditional probabilities
    """
    gts = gts[:, :, np.newaxis]
    preds = preds[:, np.newaxis, :]
    if weights is not None:
        weights = weights[:, np.newaxis, np.newaxis]
        mat = (preds * gts * weights).sum(axis=0) / weights.sum(axis=0)
    else:
        mat = (preds * gts).mean(axis=0)

    if normalize:
        mat = mat / (mat.sum(axis=norm_axis, keepdims=True) + eps)
    return mat

conditional_matrix = calc_conditional_prob(all_oof, all_true)

import matplotlib.pyplot as plt
import seaborn as sns

class_names = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']

plt.figure(figsize=(10, 8))
sns.heatmap(conditional_matrix, annot=True, cmap='Blues', fmt=".2f", xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
plt.title('EfficientNet: Conditional Probability Matrix', fontsize=16)
plt.xlabel('Predicted Classes', fontsize=14)
plt.ylabel('Actual Classes', fontsize=14)

# Increase tick label size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()

temp_true = pd.DataFrame(all_true)
temp_true.columns = class_names
mean_probabilities = temp_true.mean()

# Display the mean probabilities
print(mean_probabilities)

# Plotting a stacked bar chart
ax = mean_probabilities.plot(kind='bar', color='skyblue')
plt.title('Average Probability Distribution Across Classes', fontsize=16)
plt.ylabel('Average Probability', fontsize=14)
plt.xlabel('Classes', fontsize=14)

ax.set_ylim(0, max(mean_probabilities) + 0.05)  # Increase the limit to max value plus 0.5

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'),  # Format the label
                (p.get_x() + p.get_width() / 2., p.get_height()),  # Position
                ha = 'center', va = 'center',  # Center alignment
                xytext = (0, 9),  # 9 points vertical offset
                textcoords = 'offset points',
                fontsize=14)

# Increase tick label size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()

# ========================================================================
# ### Inference
# ========================================================================

del all_eegs, spectrograms
gc.collect()

test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
print('Test shape',test.shape)
test.head()

# READ ALL SPECTROGRAMS
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
files2 = os.listdir(PATH2)
print(f'There are {len(files2)} test spectrogram parquets')

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 100 == 0:
        print(i, ', ',end='')
    tmp = pd.read_parquet(f'{PATH2}{f}')
    name = int(f.split('.')[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values

# RENAME FOR DATALOADER
test = test.rename({'spectrogram_id': 'spec_id'}, axis=1)

import pywt, librosa

USE_WAVELET = None

NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]


# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')

    return ret


def spectrogram_from_eeg(parquet_path, display=False):

    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype='float32')

    if display: plt.figure(figsize=(10,7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256,
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40
            img[:,:,k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0

        if display:
            plt.subplot(2,2,k+1)
            plt.imshow(img[:,:,k],aspect='auto',origin='lower')
            plt.title(f'EEG {eeg_id} - Spectrogram {NAMES[k]}')

    if display:
        plt.show()
        plt.figure(figsize=(10,5))
        offset = 0
        for k in range(4):
            if k>0: offset -= signals[3-k].min()
            plt.plot(range(10_000),signals[k]+offset,label=NAMES[3-k])
            offset += signals[3-k].max()
        plt.legend()
        plt.title(f'EEG {eeg_id} Signals')
        plt.show()
        print(); print('#'*25); print()

    return img

# READ ALL EEG SPECTROGRAMS
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/'
DISPLAY = 1
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print('Converting Test EEG to Spectrograms...'); print()
for i, eeg_id in enumerate(EEG_IDS2):

    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY)
    all_eegs2[eeg_id] = img

# INFER EFFICIENTNET ON TEST
preds = []
test_ds = EEGDataset(test, mode='test', specs=spectrograms2, eeg_specs=all_eegs2)
test_loader = DataLoader(test_ds, shuffle=False, batch_size=64, num_workers=3)

for i in range(5):
    print('#'*25)
    print(f'### Testing Fold {i+1}')

    ckpt_file = f'EffNet_v{VER}_f{i}.ckpt' if LOAD_MODELS_FROM is None else f'{LOAD_MODELS_FROM}/EffNet_v{VER}_f{i}.ckpt'
    model = EEGEffnetB0.load_from_checkpoint(ckpt_file)
    model.to(device).eval()
    fold_preds = []

    with torch.inference_mode():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            pred = torch.softmax(model(test_batch), dim=1).cpu().numpy()
            fold_preds.append(pred)
        fold_preds = np.concatenate(fold_preds)

    preds.append(fold_preds)

efficientnet_predictions = np.mean(preds,axis=0)
print()
print('Test preds shape',efficientnet_predictions.shape)

# ========================================================================
# ## Model 2 - WaveNet
# ========================================================================

# ========================================================================
# ### Config and utils
# ========================================================================

class config:
    BATCH_SIZE_TEST = 32
    NUM_WORKERS = 0 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"


model_weights = [x for x in glob("/kaggle/input/hms-wavenet/*.pth")]
model_weights

def eeg_from_parquet(parquet_path: str) -> np.ndarray:
    """
    This function reads a parquet file and extracts the middle 50 seconds of readings. Then it fills NaN values
    with the mean value (ignoring NaNs).
    :param parquet_path: path to parquet file.
    :param display: whether to display EEG plots or not.
    :return data: np.array of shape  (time_steps, eeg_features) -> (10_000, 8)
    """
    # === Extract middle 50 seconds ===
    eeg = pd.read_parquet(parquet_path, columns=eeg_features)
    rows = len(eeg)
    offset = (rows - 10_000) // 2 # 50 * 200 = 10_000
    eeg = eeg.iloc[offset:offset+10_000] # middle 50 seconds, has the same amount of readings to left and right
    # === Convert to numpy ===
    data = np.zeros((10_000, len(eeg_features))) # create placeholder of same shape with zeros
    for index, feature in enumerate(eeg_features):
        x = eeg[feature].values.astype('float32') # convert to float32
        mean = np.nanmean(x) # arithmetic mean along the specified axis, ignoring NaNs
        nan_percentage = np.isnan(x).mean() # percentage of NaN values in feature
        # === Fill nan values ===
        if nan_percentage < 1: # if some values are nan, but not all
            x = np.nan_to_num(x, nan=mean)
        else: # if all values are nan
            x[:] = 0
        data[:, index] = x

    return data


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sep():
    print("-"*100)


target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other':5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)

# ========================================================================
# ### Data loading
# ========================================================================

# Prepare raw EEG

test_df = pd.read_csv(paths.TEST_CSV)
eeg_parquet_paths = glob(paths.TEST_EEGS + "*.parquet")
eeg_df = pd.read_parquet(eeg_parquet_paths[0])
eeg_features = eeg_df.columns
print(f'There are {len(eeg_features)} raw eeg features')
print(list(eeg_features))
eeg_features = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
feature_to_index = {x:y for x,y in zip(eeg_features, range(len(eeg_features)))}

CREATE_EEGS = False
all_eegs = {}
visualize = 1
eeg_paths = glob(paths.TEST_EEGS + "*.parquet")
eeg_ids = test_df.eeg_id.unique()

for i, eeg_id in tqdm(enumerate(eeg_ids)):
    # Save EEG to Python dictionary of numpy arrays
    eeg_path = paths.TEST_EEGS + str(eeg_id) + ".parquet"
    data = eeg_from_parquet(eeg_path)
    all_eegs[eeg_id] = data

# ========================================================================
# ### Butter low-pass filter
# ========================================================================

from scipy.signal import butter, lfilter

def butter_lowpass_filter(data, cutoff_freq: int = 20, sampling_rate: int = 200, order: int = 4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

# ========================================================================
# ### Dataset
# ========================================================================

class CustomDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, config,
        eegs: Dict[int, np.ndarray] = all_eegs, downsample: int = 5
    ):
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE_TEST
        self.eegs = eegs
        self.downsample = downsample

    def __len__(self):
        """
        Length of dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Get one item.
        """
        X = self.__data_generation(index)
        X = X[::self.downsample, :]
        output = {
            "X": torch.tensor(X, dtype=torch.float32)
        }
        return output

    def __data_generation(self, index):
        row = self.df.iloc[index]
        X = np.zeros((10_000, 8), dtype='float32')
        data = self.eegs[row.eeg_id]

        # === Feature engineering ===
        X[:,0] = data[:,feature_to_index['Fp1']] - data[:,feature_to_index['T3']]
        X[:,1] = data[:,feature_to_index['T3']] - data[:,feature_to_index['O1']]

        X[:,2] = data[:,feature_to_index['Fp1']] - data[:,feature_to_index['C3']]
        X[:,3] = data[:,feature_to_index['C3']] - data[:,feature_to_index['O1']]

        X[:,4] = data[:,feature_to_index['Fp2']] - data[:,feature_to_index['C4']]
        X[:,5] = data[:,feature_to_index['C4']] - data[:,feature_to_index['O2']]

        X[:,6] = data[:,feature_to_index['Fp2']] - data[:,feature_to_index['T4']]
        X[:,7] = data[:,feature_to_index['T4']] - data[:,feature_to_index['O2']]

        # === Standarize ===
        X = np.clip(X,-1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0

        # === Butter Low-pass Filter ===
        X = butter_lowpass_filter(X)

        return X

# ========================================================================
# ### DataLoader
# ========================================================================

test_dataset = CustomDataset(test_df, config)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE_TEST,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=False
)
output = test_dataset[0]
X = output["X"]
print(f"X shape: {X.shape}")

# ========================================================================
# ### Model
# ========================================================================

class Wave_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: int, kernel_size: int = 3):
        """
        WaveNet building block.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param dilation_rates: how many levels of dilations are used.
        :param kernel_size: size of the convolving kernel.
        """
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True))

        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True))

        for i in range(len(self.convs)):
            nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.convs[i].bias)

        for i in range(len(self.filter_convs)):
            nn.init.xavier_uniform_(self.filter_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.filter_convs[i].bias)

        for i in range(len(self.gate_convs)):
            nn.init.xavier_uniform_(self.gate_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.gate_convs[i].bias)

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            tanh_out = torch.tanh(self.filter_convs[i](x))
            sigmoid_out = torch.sigmoid(self.gate_convs[i](x))
            x = tanh_out * sigmoid_out
            x = self.convs[i + 1](x)
            res = res + x
        return res

class WaveNet(nn.Module):
    def __init__(self, input_channels: int = 1, kernel_size: int = 3):
        super(WaveNet, self).__init__()
        self.model = nn.Sequential(
                Wave_Block(input_channels, 8, 12, kernel_size),
                Wave_Block(8, 16, 8, kernel_size),
                Wave_Block(16, 32, 4, kernel_size),
                Wave_Block(32, 64, 1, kernel_size)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        output = self.model(x)
        return output


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model = WaveNet()
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = 0.0
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 6)
        )

    def forward(self, x: torch.Tensor):
        """
        Forwward pass.
        """
        x1 = self.model(x[:, :, 0:1])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze(dim=2)
        x2 = self.model(x[:, :, 1:2])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze(dim=2)
        z1 = torch.mean(torch.stack([x1, x2]), dim=0)

        x1 = self.model(x[:, :, 2:3])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze(dim=2)
        x2 = self.model(x[:, :, 3:4])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze(dim=2)
        z2 = torch.mean(torch.stack([x1, x2]), dim=0)

        x1 = self.model(x[:, :, 4:5])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze(dim=2)
        x2 = self.model(x[:, :, 5:6])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze(dim=2)
        z3 = torch.mean(torch.stack([x1, x2]), dim=0)

        x1 = self.model(x[:, :, 6:7])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze(dim=2)
        x2 = self.model(x[:, :, 7:8])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze(dim=2)
        z4 = torch.mean(torch.stack([x1, x2]), dim=0)

        y = torch.cat([z1, z2, z3, z4], dim=1)
        y = self.head(y)

        return y

model = CustomModel()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# ========================================================================
# ### Inference
# ========================================================================

def inference_function(test_loader, model, device):
    model.eval() # set model in evaluation mode
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, batch in enumerate(tqdm_test_loader):
            X = batch.pop("X").to(device) # send inputs to `device`
            batch_size = X.size(0)
            with torch.no_grad():
                y_preds = model(X) # forward propagation pass
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy()) # save predictions

    prediction_dict["predictions"] = np.concatenate(preds) # np.array() of shape (fold_size, target_cols)
    return prediction_dict

predictions = []

for model_weight in model_weights:
    test_dataset = CustomDataset(test_df, config)
    train_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    model = CustomModel()
    checkpoint = torch.load(model_weight)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    prediction_dict = inference_function(test_loader, model, device)
    predictions.append(prediction_dict["predictions"])
    torch.cuda.empty_cache()
    gc.collect()

wavenet_predictions = np.array(predictions)
wavenet_predictions = np.mean(wavenet_predictions, axis=0)

# ========================================================================
# ## Ensemble - Submission
# ========================================================================

# Calculate weighted average of predictions and submit
sub = pd.DataFrame({'eeg_id': test.eeg_id.values})
sub[TARGETS] = 0.75*efficientnet_predictions + 0.25*wavenet_predictions
sub.to_csv('submission.csv',index=False)
print('Submissionn shape',sub.shape)
sub.head()

# SANITY CHECK TO CONFIRM PREDICTIONS SUM TO ONE
sub.iloc[:,-6:].sum(axis=1)

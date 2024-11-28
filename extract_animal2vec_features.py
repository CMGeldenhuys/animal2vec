from typing import Dict, Tuple, Type
import warnings

import audioread
from tqdm.auto import tqdm
from argparse import ArgumentParser

import os
from glob import glob

import numpy as np

import nn
import torch
import librosa
import numpy as np
from fairseq import checkpoint_utils

def load_model(model_checkpoint):
    print(f'Loading model from {model_checkpoint}')
    models, _ = checkpoint_utils.load_model_ensemble([model_checkpoint])
    return models[0]


def get_datafiles(data_dir):
    return glob(os.path.join(data_dir, '***.wav'), recursive=True)

_SAMPLE_RATE = 8000 # model sample rate
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-checkpoint')
    parser.add_argument('--data-dir')
    parser.add_argument('--output')
    parser.add_argument('--device')
    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    data_dir = args.data_dir
    output = args.output
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    assert os.path.exists(output)

    model = load_model(model_checkpoint)
    print(f'moving model to {device}')
    model = model.to(device)
    model = model.eval()

    data_files = get_datafiles(data_dir)

    for file_path in tqdm(data_files):
        basename = os.path.basename(file_path)
        basename, *_ = basename.split('.')

        audio, sr = librosa.load(file_path, sr=_SAMPLE_RATE)
        assert sr == _SAMPLE_RATE, f'file sample rate does not match model {sr} != {_SAMPLE_RATE}'

        # Convert to tensor and add empty channel dim
        audio = torch.tensor(audio).unsqueeze(0)
        assert audio.ndim == 2
        # shape: (nchannel, nsamples)

        with torch.inference_mode():
            audio = audio.to(device)
            features = model.extract_features(source=audio)
        save_path = os.path.join(output, basename+'.pt')
        assert not os.path.exists(save_path)
        torch.save(features, save_path)

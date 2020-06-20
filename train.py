#!/usr/bin/python

# Copyright 2019 David Kügler, Technical University of Darmstadt, Darmstadt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#import the required libraries

import os
import sys
import math
import itertools

import os
import json
from argparse import ArgumentParser, ArgumentTypeError
from datetime import datetime

import sys
folder = os.path.normpath(os.path.realpath(sys.path[0]) + "/code")
sys.path.append(folder)

from argparse_helper import *

argument_parser = ArgumentParser(
    description="Train i3PosNet",
    fromfile_prefix_chars='@'
    )
    
mgroup = argument_parser.add_mutually_exclusive_group(required=True)

def tolist(x):
    return [unsigned(x)]

mgroup.add_argument(
    '--gpu',
    dest="gpus",
    type=tolist,
    help="Specify the gpu to use (defaults to gpu:0)")

mgroup.add_argument(
    '--multi_gpu',
    dest="gpus",
    nargs="+",
    type=unsigned,
    help="Activate multi-gpu training and specify the gpus"
    )
argument_parser.set_defaults(gpus=[0])

argument_parser.add_argument(
    '--prefix',
    type=str,
    default="i3PosNet_"
    help="Prefix to use for naming model output files (default: 'i3PosNet_'")

argument_parser.add_argument(
    '--batch_size',
    type=unsigned,
    default=512,
    help="Batch size, for multi-gpu-training, this should be scaled (default: 512, this should work for 8GB VRAM).")
    
argument_parser.add_argument(
    '--epochs',
    type=unsigned,
    default=80,
    help="Number of epochs to train (default: 80).")
    
argument_parser.add_argument(
    '--snapshot_interval',
    type=unsigned,
    default=1e10,
    help="interval to store snapshots of (default: Infinity = functionally off).")
    
argument_parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help="Path to the base output folder. This folder is expected to have sub-folders `snapshots`, `hyperparam` and `history`.")

argument_parser.add_argument('-d',
    '--training_data',
    type=str,
    nargs="+",
    required=True,
    help="Text file listing all HDF5 Dataset to use for training.")

argument_parser.add_argument('-d',
    '--validation_data',
    type=str,
    nargs="+",
    required=True,
    help="Text file listing all HDF5 Dataset to use for validation.")

argument_parser.add_argument(
    '--hdf_dir',
    type=str,
    default=os.get_cwd(),
    help="Path to the folder to store the hdf dataset.")


if __main__ == "__name__":

    # set the rng to defined settings
    # from keras_impl import seedsettings
    # 
    from trainkeras import TrainKeras as TrainLib

    trainer = TrainLib()
    
    args = argument_parser.parse_args()
​

    trainer["TrainingDirectory"] = args.data_dir

    # prefix for all files
    trainer["Prefix"] = args.prefix

​    # hd5 files to use for training
    trainer["HDF5Directory"] = args.hdf_dir

    # load_data is also called in train(), e.g. to infer the size of the images.
    # trainer.load_data()
    # trainer["resume"] = "" # snapshot file /...

    trainer["SnapshotInterval"] = args.snapshot_interval
    # for the full model (3,3,3) size 512 per 1080 works.
    # performance gain from 384 is minimal tho
    trainer["BatchSize"] = args.batch_size
    max_epoch = args.epochs

​    trainer["LRPolicy"] = "step" # more policies are supported, see code/trainkeras.py
    trainer["BaseLR"] = 5e-3
    trainer["H5TrainingListFile"] = args.training_data
    trainer["H5TestListFile"]     = args.validation_data # TestListFile is a misnomer, this is validation
    trainer["LayerCount"] = (3,3,39
    trainer["FCLayerCount"] = (3,2)
    trainer["ConvReg"] = "BatchNormalization_Layer"
    trainer["FCReg"] = "BatchNormalization"
    trainer["Shrinking"] = "Stride"
    trainer["Optimizer"] = "Adam"
    trainer["Padding"] = "same" # if params[0][0] == 3 else "valid"
    trainer["Epsilon"] = None
    trainer["Gamma"] = 0.1 
    trainer["StepSize"] = 35
    
    # Reset the model
    trainer.set_model(None)

    # set properties
    trainer.train(max_epoch, args.gpus)

    # release GPU memory / model from gpu
    del trainer





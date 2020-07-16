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

import json
from argparse import ArgumentTypeError

def unsigned(x):
    if int(x) < 0:
        raise ArgumentTypeError("Negative values are not valid.")
    return int(x)

def add_plot_verbose(argument_parser):
    argument_parser.add_argument('-p',
        '--plot',
        action='store_true',
        default=False,
        help="Plot all pipeline steps, only recommended for debugging (default: False).")

    argument_parser.add_argument('-v',
        '--verbose',
        action='store_true',
        default=False,
        help="Use verbose output, useful for debugging (default: False)")

def add_dataset_folder(argument_parser):
    argument_parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Path to the base folder of the dataset. This folder is expected to have sub-folders `images`, `tracked` and `parameters`.")

def add_range_list_subparser(argument_parser):
    subparsers = argument_parser.add_subparsers(help='index selection')
    range_parser = subparsers.add_parser('range',
        help="Create the Dataset from a range of indices, see `range(start, stop, step)`")
    range_parser.set_defaults(mode='range')
    range_parser.add_argument(
        '--start',
        type=int,
        default=0,
        help="The index of the first drr/xray to include in the dataset (default 0)"
        )

    range_parser.add_argument(
        '--stop',
        type=int,
        help="The index of the last drr/xray to include in the dataset +1",
        required=True
        )

    range_parser.add_argument(
        '--step',
        type=int,
        default=1,
        help="The step between indices in the dataset (default: 1)"
        )
    range_parser.add_argument(
        '--exclude',
        type=json.loads,
        default=[],
        help="A json-encoded list of indices to exclude from the range")

    json_parser = subparsers.add_parser('json',
        help="Create the Dataset from a list of json-encoded indices, supports reading from a file, use '@<filename>'"
        )
    json_parser.set_defaults(mode='list')
    json_parser.add_argument('list',
        type=json.loads,
        help='Load list of indices from a json-encoded list'
    )

def add_instrument_options(argument_parser):
    mgroup = argument_parser.add_mutually_exclusive_group(required=True)

    mgroup.add_argument(
        '--end_to_end',
        action='store_true',
        default=False,
        help="Use direct/explicit angle annotation")

    mgroup.add_argument(
        '--instrument',
        choices=["screw", "drill", "robot"],
        default="screw",
        help="The instrument to use, this initializes the positioning of screws "
        )
        
    mgroup.add_argument(
        '--custom_landmarks',
        type=json.loads,
        default=None,
        help="Use a custom pattern of landmarks, expecting a json-encoded list of float pairs.")
    argument_parser.add_argument(
        '--center_offset',
        type=json.loads,
        default=(-10, 0),
        help="How much to offset the center of the instrument (defaults to instruments defaults, e.g. (-10, 0) for screw.")

def add_samples(argument_parser, default=20):
    argument_parser.add_argument(
        '--num_samples',
        type=unsigned,
        default=default,
        help="Number of samples per image/ independent deviations from actual pose (default: {:d})".format(default))

def add_interpolation(argument_parser):
    argument_parser.add_argument(
        '--interpolation_order',
        type=unsigned,
        default=1,
        help="Order of the image interpolation (default: 1)")

def exclusion(iterator, exclusion):
    return [i for i in iterator if i not in exclusion]

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    datapath = os.path.join(opt['datapath'], 'RACE')
    dirpath = opt['datapath']
    version = None

    if not build_data.built(datapath, version_string=version):
        print('[building data: ' + datapath + ']')
        if build_data.built(datapath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(datapath)

        # Download the data.
        fname = 'RACE.tar.gz'
        url = 'http://www.cs.cmu.edu/~glai1/data/race/' + fname
        build_data.download(url, dirpath, fname)
        build_data.untar(dirpath, fname)

        # Mark the data as built.
        build_data.mark_done(datapath, version_string=version)

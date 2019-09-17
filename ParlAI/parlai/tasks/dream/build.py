#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    datapath = os.path.join(opt['datapath'], 'DREAM')
    build_data.make_dir(datapath)
    version = None

    if not build_data.built(datapath, version_string=version):
        print('[building data: ' + datapath + ']')
        if build_data.built(datapath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(datapath)

        # Download the data.
        splits = ['train', 'dev', 'test']
        for split in splits:
            fname = split + '.json'
            url = 'https://raw.githubusercontent.com/nlpdata/dream/master/data/' + fname
            build_data.download(url, datapath, fname)

        # Mark the data as built.
        build_data.mark_done(datapath, version_string=version)

#!/usr/bin/env bash

pushd /tmp
git clone https://github.com/facebookresearch/fastText.git

cd fastText
pip3 install .

cd ..
rm -rf fastText

popd

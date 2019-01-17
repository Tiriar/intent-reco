#!/usr/bin/env bash

# install sent2vec

pushd /tmp
git clone https://github.com/epfml/sent2vec.git

cd sent2vec
make

cd src
python3 setup.py build_ext
pip3 install .

cd ../..
rm -rf sent2vec

popd
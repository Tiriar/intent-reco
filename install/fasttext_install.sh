#!/usr/bin/env bash

# install FastText

pushd /tmp
git clone https://github.com/facebookresearch/fastText.git
#git checkout 5bf8b4c615b6308d76ad39a5a50fa6c4174113ea

cd fastText
pip3 install .

cd ..
rm -rf fastText

popd
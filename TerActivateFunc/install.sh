#!/bin/bash
HOME=$(pwd)
# cd $HOME/src/cpp
# echo "Installing cpp extension..."
# python setup.py clean && python setup.py install

cd $HOME/src/cuda
echo "Installing cuda extension..."
python setup.py clean && python setup.py install

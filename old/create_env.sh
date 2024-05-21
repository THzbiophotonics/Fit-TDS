#!/bin/bash
sudo apt install build-essential libssl-dev libffi-dev python3-dev
sudo apt install -y python3-venv
python3 -m venv py3-env
source ./py3-env/bin/activate
pip install --upgrade pip
py3-env/bin/pip install wheel
py3-env/bin/pip install -r requirements.txt
py3-env/bin/pip install git+https://github.com/madebr/pyOpt

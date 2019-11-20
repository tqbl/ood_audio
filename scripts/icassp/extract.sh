#!/bin/bash

set -e

python ood_audio/main.py extract test
python ood_audio/main.py extract training

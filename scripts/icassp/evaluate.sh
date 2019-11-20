#!/bin/bash

set -e

SEEDS=(1000 2000 3000 4000 5000)

evaluate() {
  echo "Evaluating '$1'..."
  ids=${SEEDS[@]/#/$1/seed=}
  python ood_audio/main.py evaluate --training_id ${ids[@]}
  echo
}

evaluate 'vgg_default'
evaluate 'vgg_clean'
evaluate 'vgg_clean-da'
evaluate 'vgg_relabel'

evaluate 'densenet_default'
evaluate 'densenet_clean'
evaluate 'densenet_clean-da'
evaluate 'densenet_relabel'

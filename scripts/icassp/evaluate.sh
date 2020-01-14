#!/bin/bash

set -e

SEEDS=(1000 2000 3000 4000 5000)

evaluate() {
  echo "Evaluating '$1'..."
  ids=${SEEDS[@]/#/$1/seed=}
  python ood_audio/main.py evaluate --training_id ${ids[@]}
  echo
}

evaluate_systems() {
  evaluate "$1_default"
  evaluate "$1_clean"
  evaluate "$1_clean-da"
  evaluate "$1_relabel"
  evaluate "$1_relabel-odin"
}

evaluate_systems vgg
evaluate_systems densenet

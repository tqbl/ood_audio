#!/bin/bash

set -e

SEEDS=(1000 2000 3000 4000 5000)

trial() {
  fname=$(basename "$1")
  id="$2_${fname%.*}/seed=$3"

  echo -e "\nTraining for '$id'..."
  python ood_audio/main.py -f "$1" train --model $2 --seed $3 --training_id "$id"

  echo -e "\nPredicting for '$id'..."
  python ood_audio/main.py -f "$1" predict test --training_id "$id" --clean=True
}

experiment() {
  echo "Running experiment for $(basename $1)..."
  for seed in ${SEEDS[@]}; do
    trial "$1" $2 $seed
  done
  echo
}

run1() {
  experiment 'default.conf' $1
  experiment 'scripts/icassp/clean.conf' $1
  experiment 'scripts/icassp/clean-da.conf' $1
}

run2() {
  experiment 'scripts/icassp/relabel.conf' $1
  experiment 'scripts/icassp/relabel-odin.conf' $1
}

run1 'vgg'
run1 'densenet'

echo 'Computing pseudo-labels...'
python ood_audio/main.py \
  -f 'scripts/icassp/clean-da.conf' --prediction_path 'metadata/_pseudo' \
  predict training --training_id 'densenet_clean-da/seed=1000' --clean=True
echo

echo 'Computing pseudo-labels (ODIN)...'
python ood_audio/main.py \
  -f 'scripts/icassp/clean-da.conf' --prediction_path 'metadata/_pseudo' \
  predict training --training_id 'densenet_clean-da/seed=1000' --clean=True --odin=True
echo

run2 'vgg'
run2 'densenet'

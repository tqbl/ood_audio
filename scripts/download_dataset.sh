#!/bin/bash

set -e

command -v curl >/dev/null 2>&1 || { echo 'curl is missing' >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo 'unzip is missing' >&2; exit 1; }

mkdir -p _dataset && cd _dataset

curl -fOL 'https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_test.zip'
curl -fOL 'https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_train.zip'

unzip 'FSDnoisy18k.audio_test.zip'
unzip 'FSDnoisy18k.audio_train.zip'

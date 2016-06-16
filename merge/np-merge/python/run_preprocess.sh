#!/bin/bash

cd "$(dirname "$0")"

echo python ./preprocess.py \
    --force \
    --block $1 $2 $3 \

python ./preprocess.py \
    --force \
    --block $1 $2 $3 \
    


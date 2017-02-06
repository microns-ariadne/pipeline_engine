#!/bin/bash

cd "$(dirname "$0")"

./build_pipeline/main -sb 1 $1 $2

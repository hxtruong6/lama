#!/usr/bin/bash
python3 bin/api.py \
    refine=True \
    model.path=$(pwd)/LaMa_models/big-lama-with-discr \
    indir=$(pwd)/LaMa_test_images2 outdir=$(pwd)/output
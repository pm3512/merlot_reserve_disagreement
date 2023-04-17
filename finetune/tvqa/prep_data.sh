#!/usr/bin/env bash

export NUM_FOLDS=128
export NUM_FOLDS_VAL=16
# Training
mkdir -p logs

# if [ $(hostname) == "shoob" ]; then
parallel -j $(nproc --all) --will-cite "python3 prep_data_siq.py -fold {1} -num_folds ${NUM_FOLDS} " ::: $(seq 0 $((${NUM_FOLDS}-1)))
parallel -j $(nproc --all) --will-cite "python3 prep_data_siq.py -fold {1} -num_folds ${NUM_FOLDS_VAL} -split=val " ::: $(seq 0 $((${NUM_FOLDS_VAL}-1)))
    #parallel -j $(nproc --all) --will-cite "python prep_data.py -fold {1} -num_folds ${NUM_FOLDS_VAL} -split=test > logs/testlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS_VAL}-1)))
# fi


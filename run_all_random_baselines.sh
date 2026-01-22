#!/bin/bash

# Master script to run all random baselines sequentially

echo "#########################################################"
echo "STARTING LAYER EA RANDOM BASELINES"
echo "#########################################################"
bash run_random_baseline_layer_batch.sh

echo ""
echo "#########################################################"
echo "STARTING GLOBAL EA RANDOM BASELINES"
echo "#########################################################"
bash run_random_baseline_global_batch.sh

echo ""
echo "#########################################################"
echo "ALL RANDOM BASELINE TASKS COMPLETED"
echo "#########################################################"

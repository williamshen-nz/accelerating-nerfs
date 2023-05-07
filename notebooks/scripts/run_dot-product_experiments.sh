#!/bin/bash
cd /home/workspace/notebooks/designs/dot-product-DUDU/
pwd
timeloop-model arch/*.yaml map/*.yaml prob/avg.dot-product.prob.yaml -o output
echo 'Experiment - Dense Baseline'
tail -15 output/timeloop-model.stats.txt


cd /home/workspace/notebooks/designs/dot-product-SCDU/
pwd
timeloop-model arch/*.yaml components/* map/*.yaml prob/avg.dot-product.prob.yaml sparse-opt/*.yaml -o output/
echo 'Experiment - SCDU Sparsity'
tail -15 output/timeloop-model.stats.txt


cd /home/workspace/notebooks/designs/dot-product-SUDU/
pwd
timeloop-model arch/*.yaml map/*.yaml prob/avg.dot-product.prob.yaml -o output
echo 'Experiment - SUDU Default'
tail -15 output/timeloop-model.stats.txt
echo 'Experiment - SUDU No Optimization'
tail -15 output/no-optimization/timeloop-model.stats.txt
echo 'Experiment - SUDU Gaiting'
tail -15 output/gating/timeloop-model.stats.txt

# Runs timeloop on Dense dot product, SCDU dot product, and SUDU dot product
# Changed Prob file (Added sapristy number), Arch (8x sizes), and Maps (to K=128)

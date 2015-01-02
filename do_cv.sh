#!/bin/sh
# $1 : deep_config.sh or benchmark_config.sh
# $2 : DBN, RBM , etc for deep_config.sh
source $1 $2

echo "CV"
echo $cv_results_dir
cross_validation $cv_data_dir $cv_arff_file $cv_results_dir

#!/bin/sh
source benchmark_config.sh
# first step : generate benchmark features
compute_benchmark_features $DATA_PATH/$folder
# second step : generate cv
merge_arff $bench_folder/train_benchmark_$bench.arff $bench_folder/test_benchmark_$bench.arff $bench_folder/benchmark_$bench.arff
cd $bench_folder
mkdir -p cv
generate_cv benchmark_$bench.arff 10

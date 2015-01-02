source folder.sh
bench=VectSum
bench_folder=$DATA_PATH/data/$folder/Benchmark

train=../$bench_folder/train_benchmark_$bench.arff 
test=../$bench_folder/test_benchmark_$bench.arff
test_raw=$DATA_PATH/$folder/test.arff
results_dir=boosting_results/results_benchmark

cv_data_dir=$DATA_PATH/$folder/Benchmark/cv
cv_arff_file=benchmark_$bench.arff
cv_results_dir=boosting_results/cv_results_benchmark

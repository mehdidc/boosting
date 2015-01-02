source folder.sh
bench=VectSum
bench_folder=../data/$folder/Benchmark

train=../$bench_folder/train_benchmark_$bench.arff 
test=../$bench_folder/test_benchmark_$bench.arff
test_raw=../../data/$folder/test.arff
results_dir=results_benchmark

cv_data_dir=../data/$folder/Benchmark/cv
cv_arff_file=benchmark_$bench.arff
cv_results_dir=cv_results_benchmark

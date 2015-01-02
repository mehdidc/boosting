#!/bin/sh
#$1 : deep_config.sh or benchmark_config.sh
#$2 : DBN or RBM,etc for deep_config.sh
source $1 $2
#step 1 : normal train
echo "TRAIN"
cd $results_dir
train $train $test
min_it=`test_errors results.dta | grep argmin | awk '{print $2}'`
test_errors results.dta
hashes  $test $test_raw > hash_mapping
posteriors model.xml $test posteriors_output $min_it
posteriors_to_errors posteriors_output $test > hash_errors_new_features
mapto hash_errors_new_features hash_mapping > hash_errors
indexes_from_hashes $test_raw hash_errors > index_errors
cd ..

#!/bin/sh
source folder.sh
method=$1
compute_deep_features ../data/$folder $method
python2.7 /exp/appstat/cherti/Projects/Boosting/PythonLib/ILC_experiment.py process-filters --arff-directory ../data/$folder ../data/$folder/$method
merge_arff ../data/$folder/$method/train_$method.arff ../data/$folder/$method/test_$method.arff ../data/$folder/$method/$method.arff
cd ../data/$folder/$method
generate_cv $method.arff 10
cd -

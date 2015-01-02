#!/bin/sh
source folder.sh
method=$1
compute_deep_features $DATA_PATH/$folder $method
python2.7 /exp/appstat/cherti/Projects/Boosting/PythonLib/ILC_experiment.py process-filters --arff-directory $DATA_PATH/$folder $DATA_PATH/$folder/$method
merge_arff $DATA_PATH/$folder/$method/train_$method.arff $DATA_PATH/$folder/$method/test_$method.arff $DATA_PATH/$folder/$method/$method.arff
cd $DATA_PATH/$folder/$method
generate_cv $method.arff 10
cd -

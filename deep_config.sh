source folder.sh

deep_folder=$DATA_PATH/$folder/$1

train=../$deep_folder/train_$1.arff 
test=../$deep_folder/test_$1.arff
test_raw=$DATA_PATH/$folder/test.arff
results_dir=boosting_results/results_$1

cv_data_dir=$DATA_PATH/$folder/$1/cv
cv_arff_file=$1.arff
cv_results_dir=boosting_results/cv_results_$1

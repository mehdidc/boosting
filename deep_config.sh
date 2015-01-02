source folder.sh

deep_folder=../data/$folder/$1

train=../$deep_folder/train_$1.arff 
test=../$deep_folder/test_$1.arff
test_raw=../../data/$folder/test.arff
results_dir=results_$1

cv_data_dir=../data/$folder/$1/cv
cv_arff_file=$1.arff
cv_results_dir=cv_results_$1

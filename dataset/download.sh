dataset=$1
source activate torch
# create dir
mkdir $dataset
cd $dataset
# download
axel -n 12 -a http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_$dataset.json.gz
axel -n 12 -a http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_$dataset.json.gz
axel -n 12 -a http://snap.stanford.edu/data/amazon/productGraph/image_features/categoryFiles/image_features_$dataset.b

# unzip
gunzip meta_$dataset.json.gz
gunzip reviews_$dataset.json.gz

# exit
cd ..
python preprocess.py --dataset $dataset

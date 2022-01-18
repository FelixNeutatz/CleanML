python mislabels/credit_mislabels.py -mislabel_percent 0.05 -clean_validation_labels False
python mislabels/eeg_mislabels.py -mislabel_percent 0.05 -clean_validation_labels False
python mislabels/us_census_mislabels.py -mislabel_percent 0.05 -clean_validation_labels False

python mislabels/credit_mislabels.py -mislabel_percent 0.1 -clean_validation_labels False
python mislabels/eeg_mislabels.py -mislabel_percent 0.1 -clean_validation_labels False
python mislabels/us_census_mislabels.py -mislabel_percent 0.1 -clean_validation_labels False

python mislabels/credit_mislabels.py -mislabel_percent 0.05 -clean_validation_labels True
python mislabels/eeg_mislabels.py -mislabel_percent 0.05 -clean_validation_labels True
python mislabels/us_census_mislabels.py -mislabel_percent 0.05 -clean_validation_labels True

python mislabels/credit_mislabels.py -mislabel_percent 0.1 -clean_validation_labels True
python mislabels/eeg_mislabels.py -mislabel_percent 0.1 -clean_validation_labels True
python mislabels/us_census_mislabels.py -mislabel_percent 0.1 -clean_validation_labels True

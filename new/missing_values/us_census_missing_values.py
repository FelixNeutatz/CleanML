import os
from new.utils import run

if __name__ == "__main__":
    target_label = "Income"
    drop_labels = []
    dirty_path = '/home/neutatz/Software/CleanML/data/USCensus/missing_values/dirty_train.csv'
    clean_path = '/home/neutatz/Software/CleanML/data/USCensus/missing_values/impute_holoclean_train.csv'
    save_path = '/home/neutatz/data/cleanml_results/'
    file_name = save_path + os.path.basename(__file__)

    run(clean_path, dirty_path, target_label, drop_labels, file_name=file_name)



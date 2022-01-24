import os
from new.utils import run

if __name__ == "__main__":
    target_label =  "SeriousDlqin2yrs"
    drop_labels = []
    dirty_path = '/home/neutatz/Software/CleanML/data/Credit/outliers/dirty_train.csv'
    clean_path = '/home/neutatz/Software/CleanML/data/Credit/outliers/clean_HC_impute_holoclean_train.csv'
    save_path = '/home/neutatz/data/cleanml_results/'
    file_name = save_path + os.path.basename(__file__) + '_autoclean_'

    run(clean_path, dirty_path, target_label, drop_labels, file_name=file_name, avoid_clean_eval=True)



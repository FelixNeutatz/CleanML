import os
from new.utils import run

if __name__ == "__main__":
    target_label = "Survived"
    drop_labels = ['PassengerId', 'Name']
    dirty_path = '/home/neutatz/Software/CleanML/data/Titanic/raw/raw.csv'
    clean_path = '/home/neutatz/Software/CleanML/data/Titanic/raw/Holoclean_mv_clean.csv'
    save_path = '/home/neutatz/data/cleanml_results/'
    file_name = save_path + os.path.basename(__file__)

    run(clean_path, dirty_path, target_label, drop_labels, file_name=file_name)



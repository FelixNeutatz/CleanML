import os
from new.utils import run

if __name__ == "__main__":
    target_label =  "Sentiment"
    drop_labels = ["Date", "Unnamed: 0", "City"]
    dirty_path = '/home/neutatz/Software/CleanML/data/Company/inconsistency/dirty_train.csv'
    clean_path = '/home/neutatz/Software/CleanML/data/Company/inconsistency/clean_train.csv'
    save_path = '/home/neutatz/data/cleanml_results/'
    file_name = save_path + os.path.basename(__file__)

    run(clean_path, dirty_path, target_label, drop_labels, file_name=file_name)


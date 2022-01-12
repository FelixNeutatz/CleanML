import os
from new.utils import run

if __name__ == "__main__":
    target_label = "priceRange"
    drop_labels = ["streetAddress", "telephone", "website"]
    dirty_path = '/home/neutatz/Software/CleanML/data/Restaurant/inconsistency/dirty_train.csv'
    clean_path = '/home/neutatz/Software/CleanML/data/Restaurant/inconsistency/clean_train.csv'
    save_path = '/home/neutatz/data/cleanml_results/'
    file_name = save_path + os.path.basename(__file__)

    run(clean_path, dirty_path, target_label, drop_labels, use_autosklearn=False, file_name=file_name)



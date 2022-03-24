import pandas as pd

def remove_whitespace(my_str):
    try:
        return my_str.strip()
    except:
        return my_str

def remove_noise(my_str):
    try:
        return float("%.3f" % my_str)
    except:
        return my_str


#clean= pd.read_csv('/home/neutatz/Software/CleanML/data/USCensus/missing_values/impute_holoclean_train.csv')
#dirty = pd.read_csv('/home/neutatz/Software/CleanML/data/USCensus/missing_values/dirty_train.csv')

#clean = pd.read_csv('/home/neutatz/Software/CleanML/data/Credit/outliers/clean_HC_impute_holoclean_train.csv')
#dirty = pd.read_csv('/home/neutatz/Software/CleanML/data/Credit/outliers/dirty_train.csv')

clean = pd.read_csv('/home/neutatz/Software/CleanML/data/University/inconsistency/clean_train.csv')
dirty = pd.read_csv('/home/neutatz/Software/CleanML/data/University/inconsistency/dirty_train.csv')


for col in clean.columns:
    clean[col] = clean[col].apply(remove_whitespace).apply(remove_noise)
for col in dirty.columns:
    dirty[col] = dirty[col].apply(remove_whitespace).apply(remove_noise)

for col in clean.columns:
    print(col)
    print(clean[col].compare(dirty[col]))

print(clean[col].compare(dirty[col]))
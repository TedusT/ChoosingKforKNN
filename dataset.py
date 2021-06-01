from pathlib import Path
import glob
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle


# %% Aggregate multiple files with the same features and then encode the labels.
# If necessary, use another encoder for the categorical features.

class preprocessdata:
    def __init__(self):
        self.data_dir = Path('Data')

    def get_set(self, limit=None):
        all_files = glob.glob(str(self.data_dir) + "/*.csv")
        # all the files are aggregated in to a data frame including n samples of mappings with 8 features
        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0, delimiter=",")
            # use can user the name of files as an index
            name = os.path.basename(filename)
            df['filename'] = name.split('.')[0]
            li.append(df)
        new_df = pd.concat(li, ignore_index=True)

        new_df = new_df.dropna()
        '''the encoder uses a OneHotEncoder approach and it is pickled for the future data in the production stage'''

        # Encode labels in column 'Class'.
        label_encoder = LabelEncoder()
        new_df['Class']=label_encoder.fit_transform(new_df['Class'])
        # The name of files, if it is only one, are removed here
        new_df = new_df.iloc[:,:-1]

        # If you have a list of categorical features, use a OneHotEncoder to encode them and save it accordingly.
        # Ohe_encoder = OneHotEncoder(handle_unknown='ignore')

        # Save the encoder for persistency reasons

        filehandler = open("encoder.obj", "wb")
        pickle.dump(label_encoder, filehandler)
        filehandler.close()

        return new_df

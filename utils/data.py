# importing required libraries
import os
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split


DATA_PATH = '/Workspace/data'


class DataPipeline:
    """
    A one pass class for managing everything Data. 

    Args:
        name (str): Name of the data.
        version (str): Data version for tracking.
        holdout (bool): Returns a third split called holdout split. The default is True.
        test_size (float): The percentage of the data to be used for testing.
                            The default is 0.2.
        holdout_size (float): The percentage of the data to be used for holdout.
                                The default is 0.5. This field is optional if
                                holdout is False.
        stratify (List[str]): A list of columns to stratify on.
    """
    def __init__(
        self,
        name: str,
        version: str,
        holdout: bool = False,
        test_size: float = 0.2,
        holdout_size: float = 0.5,
        stratify: Optional[List[str]] = None):
        

        self.holdout = holdout
        self.test_size = test_size
        self.holdout_size = holdout_size
        self.stratify = stratify
        self.name = name
        self.version = version

    def read(self) -> None:
        """
        Creates a dataframe attibute within the data object.
        """
        try:    
            # TODO: Hardcoded schema. Should be made flexible later.
            self.df = pd.read_csv(os.path.join(
                DATA_PATH, self.name, 'raw', 'data.csv'))
        except Exception as e:
            raise Exception(f'Error reading file. {str(e)}')
    
    def split(self) -> None:
        """
        This is a helper function to divide the dataset into validation splits.
        The data is divided into train and test splits by default and has an option
        to get the third split called holdout split. The splits are saved in a new pandas
        column called splits which can include train, test and optionally holdout.
        """
        
        def fix_stratify(df, stratify):
            """
            This is a helper function to handle how stratification is handled in pandas dataframe.
            """
            if stratify:
                stratify = df[stratify]
            else:
                stratify = None
            return stratify
        
        try:
            # initialized holdout to be None 
            holdout = None
            # checks if the ratio of train dataset > 0. If not, we raise an exception.
            if self.holdout and (self.holdout_size + self.test_size >= 1.0):
                raise ValueError("holdout_size and test_size combined must be less than 1.0")

            # The next code uses sklearn train_test_split to split the data into train and test splits.
            # More information about the API can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
            if self.holdout:
                train_test, holdout = train_test_split(self.df,
                                                       test_size=self.holdout_size,
                                                       random_state=42,
                                                       shuffle=True,
                                                       stratify=fix_stratify(self.df, self.stratify))
                holdout['split'] = ['holdout']*len(holdout)
            else:
                train_test = self.df

            train, test = train_test_split(train_test,
                                           test_size=self.test_size,
                                           random_state=42,
                                           shuffle=True,
                                           stratify=fix_stratify(train_test, self.stratify))
            train['split'], test['split'] = ['train']*len(train), ['test']*len(test)
            self.df = pd.concat([train, test] + (
                [holdout] if holdout is not None else []))

        except Exception as e:
            raise Exception(f'Error splitting data. {str(e)}')
        
    def get_train_data(self) -> pd.DataFrame:
        """
        Get Train split
        """
        return self.df[self.df['split'] == 'train']

    def get_test_data(self):
        """
        Get test split
        """
        return self.df[self.df['split'] == 'test']
    
    def get_holdout_data(self):
        """
        Get holdout split
        """
        return self.df[self.df['split'] == 'holdout']

    #TODO: add a function to create a distribution of the data based on the splits.
    def create_split_distributions():
        raise NotImplementedError()

    #TODO: add a function to create a distribution of the data based on the target labels.
    def create_target_distributions():
        raise NotImplementedError()


def generate_and_save_data(
    name: str,
    version: str,
    holdout: bool,
    test_size: float,
    holdout_size: float,
    stratify: Optional[List[str]]):
    """
    Helper function to run DataPipeline class and generate processed data 

    Args:
        name (str): Name of the data.
        version (str): Data version for tracking.
        holdout (bool): Returns a third split called holdout split. The default is True.
        test_size (float): The percentage of the data to be used for testing.
                            The default is 0.2.
        holdout_size (float): The percentage of the data to be used for holdout.
                                The default is 0.5. This field is optional if
                                holdout is False.
        stratify (List[str]): A list of columns to stratify on.
    """
    data_obj = DataPipeline(
        name=name,
        version=version,
        holdout=holdout,
        test_size=test_size,
        holdout_size=holdout_size,
        stratify=stratify
    )
    data_obj.read()
    data_obj.split()

    if not os.path.exists(os.path.join(DATA_PATH, data_obj.name, data_obj.version)):
        os.makedirs(os.path.join(DATA_PATH, data_obj.name, data_obj.version))
    
    data_obj.df.to_csv(
        os.path.join(DATA_PATH, data_obj.name, data_obj.version, 'split.csv'))
    

def retrieve_data(name: str, version: str) -> pd.DataFrame:
    """
    Retrieve previously generated data
    """
    df = pd.read_csv(os.path.join(DATA_PATH, name, version, 'split.csv'))
    return df

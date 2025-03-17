import click
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split


class DataPipeline:
    """
    A one pass class for managing everything Data. 

    Args:
        file_path (str): The path to the file containing the data.
        holdout (bool): Returns a third split called holdout split. The default is True.
        test_size (float): The percentage of the data to be used for testing.
                            The default is 0.2.
        holdout_size (float): The percentage of the data to be used for holdout.
                                The default is 0.5. This field is optional if
                                holdout is False.
        stratify (List[str]): A list of columns to stratify on.
    """
    def __init__(self, file_path: str,
                 holdout: bool = False,
                 test_size: float = 0.2,
                 holdout_size: float = 0.5,
                 stratify: Optional[List[str]] = None):
        self.file_path = file_path
        self.holdout = holdout
        self.test_size = test_size
        self.holdout_size = holdout_size
        self.stratify = stratify


    def read(self) -> None:
        """
        Creates a dataframe attibute within the data object.
        """
        try:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            else:
                raise Exception('Unsupported file format. We only support CSV files!')
        except Exception as e:
            raise Exception(f'Error reading file. {str(e)}')
    
    def split(self) -> None:
        holdout = None
        def fix_stratify(df, stratify):
            if stratify:
                stratify = df[stratify]
            else:
                stratify = None
            return stratify
        try:
            if self.holdout and (self.holdout_size + self.test_size >= 1.0):
                raise ValueError("holdout_size and test_size combined must be less than 1.0")

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
        
    def get_train_data(self):
        return self.df[self.df['split'] == 'train']

    def get_test_data():
        return self.df[self.df['split'] == 'test']
    
    def get_holdout_data():
        return self.df[self.df['split'] == 'holdout']

    def create_distributions():
        raise NotImplementedError()


def generate_data(file_path, holdout, test_size, holdout_size, stratify):
    data_obj = Data(
        file_path=file_path,
        holdout=holdout,
        test_size=test_size,
        holdout_size=holdout_size,
        stratify=stratify
    )
    data_obj.read()
    data_obj.split()
    return data_obj

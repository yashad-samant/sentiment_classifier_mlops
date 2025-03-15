import click
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split


class Data:
    """
    A one pass class for managing everything Data. 

    Args:
        file_path (str): The path to the file containing the data.
        holdout (bool): Returns a third split called holdout split. The default is True.
        test_size (float): The percentage of the data to be used for testing. The default is 0.2.
        holdout_size (float): The percentage of the data to be used for holdout. The default is 0.5. This field is optional if hold_split is False.
        stratify (List[str]): A list of columns to stratify on.
    """
    def __init__(self, file_path: str, holdout: bool = False, test_size: float = 0.2, holdout_size: float = 0.5, stratify: Optional[List[str]] = None):
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
        try:
            if self.holdout:
                train_test, holdout = train_test_split(self.df, test_split=self.holdout_size, random_state=42, shuffle=True, stratify=stratify)
                holdout['split'] = ['holdout']*len(holdout)
            else:
                train_test = self.df

            train, test = train_test_split(train_test, test_split=self.test_size, random_state=42, shuffle=True, stratify=stratify)
            train['split'], test['split'] = ['train']*len(train), ['test']*len(test)
            
            self.df = pd.concat([train, test] + ([holdout] if holdout is not None else []))

        except Exception as e:
            raise Exception(f'Error splitting data. {str(e)}')
    
    def create_distributions():
        raise NotImplementedError()


@click.command()
@click.option('--file_path', type=str, required=True, help='The path to the file containing the data.')
@click.option('--holdout', type=bool, default=False, help='Returns a third split called holdout split. The default is True.')
@click.option('--test_split_ratio', type=float, default=0.2, help='The percentage of the data to be used for testing. The default is 0.2.')
@click.option('--holdout_split_ratio', type=float, default=0.5, help='The percentage of the data to be used for holdout. The default is 0.5. This field is optional if hold_split is False.')
@click.option('--stratify', type=List[str], default=None, help='A list of columns to stratify on.')
def generate_data(file_path, holdout, test_split_ratio, holdout_split_ratio, stratify):
    data_obj = Data(file_path, holdout, test_split_ratio, holdout_split_ratio, stratify)
    return data_obj
    


if __name__ == '__main__':
    generate_data()

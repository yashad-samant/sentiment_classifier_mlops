import pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read(self) -> pd.DataFrame:
        try:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            else:
                raise Exception('Unsupported file format. We only support CSV files!')
        except Exception as e:
            raise str(e)
    
    def split(test_split: float = 0.2, hold_split: float = 0.5, stratify: Optional[List[str] = None]) -> pd.DataFrame:
        try:
            train_test, holdout = train_test_split(test_split=hold_split, random_state=42, shuffle=True, stratify=stratify)
            train, test = train_test_split(train_test, test_split=test_split, random_state=42, shuffle=True, stratify=stratify)
            train['split'], test['split'], holdout['split'] = ['train']*len(train), ['test']*len(test), ['holdout']*len(holdout)
            self.df = pd.concat([train, test, holdout])

        except Exception as e:
            raise Exception(f'Error splitting data. {str(e)}')
    
    def create_distributions():
        raise NotImplementedError()
        


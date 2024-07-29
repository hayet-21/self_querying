from typing import List
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


RAW_DATA_PATH= os.path.abspath(f"../{os.getenv('RAW_DATA_PATH')}")
BATCH_SIZE= int(os.getenv('BATCH_SIZE'))
FEATURES= eval(os.getenv('FEATURES'))

def sample_data() -> pd.DataFrame:
    """ read the raw data file in csv formats and returns a sample of size sample_size"""
    return pd.read_csv(RAW_DATA_PATH, encoding='utf-8').sample(BATCH_SIZE, random_state=42)[FEATURES]

def save_csv(df: pd.DataFrame, save_filename: str) -> None:
    """ save the df into a csv file in a specified data location as filename"""
    save_path= f'../data/{save_filename}'
    df.to_csv(save_path, index=False)

def save_txt(content: List[str], save_filename: str) -> None:
    """ save the content, which is a list of lines, into a txt file in a specified location in save_path"""
    save_path= save_path= f'../data/{save_filename}'
    with open(save_path, 'w') as f:
        f.write('\n'.join(content))


def df2strlist(df: pd.DataFrame) -> List[str]:
    """ convert a pandas dataframe into list of strings where each line is a list item """

    columns= df.columns
    content= [' '.join([f"{col}= {row[col]}" for col in columns]) for row in df.iterrows()]
    return content

def main(csv: str|None= None, txt: str|None= None) -> None :

    if csv is None and txt is None:
        return

    df= sample_data()
    if csv is not None:
        assert csv.split('.')[-1] == 'csv', "your filename must end with '.csv'"
        save_csv(df, csv)
    if txt is not None:
        assert csv.split('.')[-1] == 'txt', "your filename must end with '.txt'"
        content= df2strlist(df)
        save_txt(content, txt)
    del df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="filename where to save the csv file", required= False)
    parser.add_argument("--txt", type=str, help="filename where to save the txt file", required= False)
    args = parser.parse_args()
    main(csv=args.csv, txt=args.txt)



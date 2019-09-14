import os
import pandas as pd

def make_split_label(x):
    if x['class_count'] != 1:
        return 0
    if str(x['1']) != 'nan':
        return 1
    if str(x['2']) != 'nan':
        return 2
    if str(x['3']) != 'nan':
        return 3
    if str(x['4']) != 'nan':
        return 4

def main():
    train_df = pd.read_csv(os.path.join('..', 'input', 'train.csv'))
    train_df['ImageId'], train_df['ClassId'] = zip(*train_df['ImageId_ClassId'].apply(lambda x: x.split('_')))
    train_df = pd.pivot_table(train_df, index='ImageId', columns='ClassId', values='EncodedPixels', aggfunc=lambda x: x, dropna=False)
    train_df = train_df.reset_index()
    train_df.columns = [str(i) for i in train_df.columns.values]
    train_df['class_count'] = train_df[['1', '2', '3', '4']].count(axis=1)
    train_df['split_label'] = train_df[['1', '2', '3', '4', 'class_count']].apply(lambda x: make_split_label(x), axis=1)
    train_df.to_csv(os.path.join('..', 'input', 'preprocessed_train.csv'), index=False)

if __name__ == '__main__':
    main()

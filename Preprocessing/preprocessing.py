import os
import pandas as pd
import streamlit as st

class PreprocessingClass:

    def delete_unnecessary_rows(df):
        real_columns = df.loc[0]
        df.columns = real_columns
        df.drop(labels=0, axis=0, inplace=True)
        df.columns=df.columns.astype(str)
        df.rename(columns={'NaT':'Date'}, inplace=True)
    
    def change_data_type(df):
        params = df.columns
        parameters = params[1:]
    
        for param in parameters:
            df[param] = df[param].astype(str).str.replace('-','0', regex=True)
            df[param] = df[param].astype(str).str.replace('.', '', regex=True)
            df[param] = df[param].astype(str).str.replace(',', '.', regex=True)
            df[param] = df[param].astype(float)
        

    def change_dataset_index(df):
        df.index = df['Date']
        df = df.resample('D').mean()


    def replace_nullvalues_with_mean(df):
        pass

    def normalize_data_values(param):
        pass

    def inverse_normalized_data_values(param):
        pass

    def convert_xlsx2csv(NEW_DATA_PATH, df_name, df):
        df.to_csv(NEW_DATA_PATH + '/' + df_name + '.csv', index=False)

    


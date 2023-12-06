#Date: 30 Nov 2023
#Author: Jatin Kadge

#Splits the one data file (.csv file) created by the Receive code into multiple csv files, each file containing 1 instance of a gesture 
#Transmission code name: Basic_data_collection_BLE_RP2040_v2.ino
#Receiver code name: RP2040-Receive_via_BLE_bleak_data_collection_v2.py

import pandas as pd
import numpy as np

file = 'C:/Users/user/Desktop/Monty/IITM/Work Activity/Codes/Python codes/Rp2040_Data5-3.3 secs/.csv'                        #Add path to a csv file //Eg. Left_Swing.csv for importing the file for splitting into sperate instance
folder = 'C:/Users/user/Desktop/Monty/IITM/Work Activity/Codes/Python codes/Rp2040_Data5-3.3 secs/-Samples'                  ##Add path to a folder //Eg. Left_Swing-Samples to store the multiple .csv file created that consists 1 instance each of the gesture

# Load the CSV file into a DataFrame
input_csv_file = file
df = pd.read_csv(input_csv_file, header = None)

df_reset = df.reset_index(drop=True)

# Find the indices where all elements in a row are 0
#zero_rows_indices = (df == 0).all(axis=1).to_numpy().nonzero()[0]
zero_rows_indices = np.where((df == 0).all(axis=1))[0]
#print(zero_rows_indices)

# Split the DataFrame based on zero rows
start_index = 0
for end_index in zero_rows_indices:
    # Create a new DataFrame for each section
    section_df = df.iloc[start_index:end_index]
    
    # Define the output file path for each section
    output_csv_file = f'{folder}/output_{start_index}_{end_index}.csv'
    
    # Save the DataFrame to the new CSV file
    section_df.to_csv(output_csv_file, index=False, header = None)
    
    # Update the start_index for the next section
    start_index = end_index + 1

# Handle the last section (after the last zero row)
if start_index < len(df):
    section_df = df.iloc[start_index:]
    output_csv_file = f'{folder}/output_{start_index}_{len(df)}.csv'
    section_df.to_csv(output_csv_file, index=False, header = None)

import pandas as pd
import numpy as np
import os
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s', filename='history.log')

# Open and read the log file
with open('history.log', 'r') as file:
    logs = file.read()
    file.close()

log_batch_num = 0
log_stop_num = -1
log_handling_num = 0

if logs:
    log_handling_num = int(logs.split('\n')[0].split(' - ')[-1].split(': ')[-1])
    log_batch_num = int(logs.split('\n')[-2].split(' - ')[-2].split(': ')[-1])
    log_stop_num = int(logs.split('\n')[-2].split(' - ')[-1].split(': ')[-1])
    handling_num_data = log_handling_num
else:
    handling_num_data = int(input("Please input the number of data per one batch: "))
    logging.info(f'Handling Number of Data: {handling_num_data}')

# folder path
files_path  = './files'
result_path = './results'

# load all columns Title
columns = np.transpose(pd.read_csv('column.csv').values)[0]
title   = pd.read_csv('title.csv')
gender  = pd.read_csv('gender.csv')
country = pd.read_csv('country.csv')
state   = pd.read_csv('state.csv')
phone   = pd.read_csv('phone.csv')
industry    = pd.read_csv('industry.csv')
emailtype   = pd.read_csv('emailtype.csv')

def find_last_filename(arr, text):
    for element in reversed(arr):
        if text in element:
            return element.split('.')[0]
    return None

def filter_category(data, category, title, path):
    # check if given path is existed
    if os.path.exists(path) == False or os.path.isdir(path) == False:
        os.mkdir(path)

    res = pd.DataFrame()

    for i in range(len(category)):
        isMatch = []

        for j in range(len(data)):
            if category['title'][i].lower() == "phone":
                if isinstance(data[title][j], str):
                    isMatch.append(True)
                else:
                    isMatch.append(False)
            elif category['title'][i].lower() == "no phone":
                if isinstance(data[title][j], str):
                    isMatch.append(False)
                else:
                    isMatch.append(True)
            else:
                if isinstance(data[title][j], str):
                    if category['title'][i].lower() in data[title][j].lower():
                        isMatch.append(True)
                    else:
                        isMatch.append(False)
                else:
                    isMatch.append(False)
        
        df_isMatch = pd.DataFrame(isMatch, columns=['isMatch'])

        res = data[df_isMatch['isMatch'] == True]
        if len(res) < 1:
            continue
        
        filename = category['title'][i]
        if '/' in filename:
            filename = filename.replace("/", " & ")

        last_filename = find_last_filename(os.listdir(path), filename)

        if last_filename is not None:
            ex_data = pd.read_csv(f'{path}/{last_filename}.csv')
            count = len(ex_data)
            left_num = 100000 - count

            if left_num > len(res):
                ex_data = pd.concat([ex_data, res])
                ex_data.to_csv(f'{path}/{last_filename}.csv', encoding='utf_8_sig', index=False)
            else:
                ex_data = pd.concat([ex_data, res.iloc[0:left_num-1]])
                ex_data.to_csv(f'{path}/{last_filename}.csv', encoding='utf_8_sig', index=False)

                res = res.iloc[left_num:]
                if '_' in last_filename:
                    n = int(last_filename.split('_')[-1]) + 1
                else:
                    n = 1

                last_filename = last_filename.split('_')[0]

                k = 1
                row_num = len(res)

                while(row_num > 100000 * k):
                    sub_res = res.iloc[100000 * (k-1) : 100000 * k - 1]
                    sub_res.to_csv(f'{path}/{last_filename}_{n}.csv', encoding='utf_8_sig', index=False)
                    k = k + 1
                    n = n + 1

                sub_res = res.iloc[100000 * (k-1) :]
                sub_res.to_csv(f'{path}/{last_filename}_{n}.csv', encoding='utf_8_sig', index=False)
        else:
            if len(res) <= 100000:
                res.to_csv(f'{path}/{filename}.csv', encoding='utf_8_sig', index=False)
            else:
                n = 1
                k = 1
                row_num = len(res)

                while(row_num > 100000 * k):
                    sub_res = res.iloc[100000 * (k-1) : 100000 * k - 1]
                    sub_res.to_csv(f'{path}/{filename}_{n}.csv', encoding='utf_8_sig', index=False)
                    k = k + 1
                    n = n + 1

                sub_res = res.iloc[100000 * (k-1) :]
                sub_res.to_csv(f'{path}/{filename}_{n}.csv', encoding='utf_8_sig', index=False)

# hyperparameters
count = 0
batch = log_batch_num

total_data = pd.DataFrame()
file_list = os.listdir(files_path)

foloder_path = os.path.join(result_path, title['title'][log_stop_num + 1])
if os.path.exists(foloder_path) and os.path.isdir(foloder_path):
    print(f'delete {foloder_path}')
    shutil.rmtree(foloder_path)

start_pos = log_handling_num * (log_batch_num - 1)

# load main datasets
for k in range(start_pos, len(file_list)):
    # Load CSV files
    data = pd.read_csv(f'{files_path}/{file_list[k]}')
    
    # filter if Email Type is Personal or Current_Professional 
    data = pd.DataFrame(data[columns])
    data = pd.concat([data[data['Email_0_Type'] == "Personal"], data[data['Email_0_Type'] == "Current_Professional"]])

    # append data 
    total_data = pd.concat([total_data, data])

    count = count + 1
    print(f'batch {batch}: {count} files are loaded!')

    if count == handling_num_data:
        data = total_data.reset_index(drop=True)

        result = pd.DataFrame()

        for i in range(log_stop_num + 1, len(title)):
            isMatch = []

            for j in range(len(data)):
                if isinstance(data['Job_Title'][j], str):
                    if all(word in data['Job_Title'][j].lower() for word in title['title'][i].lower().split(' ')):
                        isMatch.append(True)
                    else:
                        isMatch.append(False)
                else:
                    isMatch.append(False)
            
            df_isMatch = pd.DataFrame(isMatch, columns=['isMatch'])

            result = data[df_isMatch['isMatch'] == True]
            if len(result) < 1:
                continue

            result = result.reset_index(drop=True)

            filename = title['title'][i]
            filepath = os.path.join(result_path, filename)

            if os.path.exists(filepath) == False or os.path.isdir(filepath) == False:
                os.mkdir(filepath)

            path = os.path.join(filepath, 'All Countries')
            filter_category(result, country, "Location_Name", path)

            path = os.path.join(filepath, "All Industries")
            filter_category(result, industry, "Industry", path)

            path = os.path.join(filepath, "All States")
            filter_category(result, state, "Location_Name", path)

            path = os.path.join(filepath, "Email Type")
            filter_category(result, emailtype, "Email_0_Type", path)

            path = os.path.join(filepath, "Phone")
            filter_category(result, phone, "Phone_0", path)

            path = os.path.join(filepath, "Gender")
            filter_category(result, gender, "Gender", path)

            print(f"\t{filename}  completed!" )
            logging.debug(f'batch: {batch} - {filename}: {i}')

        total_data = pd.DataFrame()
        batch = batch + 1
        count = 0
        log_stop_num = 0

with open('history.log', 'w'):
    pass  # This will clear the contents of the file

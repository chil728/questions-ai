import clean_json
import json2csv
import os

def traverse_and_clean_json2csv(input_folder, temp_folder, output_folder):
    clean_json.traverse_and_clean_json(input_folder, temp_folder)
    json2csv.json_convert_csv(temp_folder, output_folder)

if __name__ == '__main__':
    num_ch = ["一級", "二級", "三級", "四級"]
    for ch in num_ch:
        data_folder = os.path.abspath(f'./data/{ch}')
        cleaned_folder = os.path.abspath(f'./cleaned_data/{ch}')
        cleaned_csv_folder = os.path.abspath(f'./cleaned_data_csv/{ch}')
        traverse_and_clean_json2csv(data_folder, cleaned_folder, cleaned_csv_folder)
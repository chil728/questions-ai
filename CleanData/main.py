import clean_json
import json2csv
import os

def traverse_and_clean_json2csv(input_folder, temp_folder, output_folder):
    clean_json.traverse_and_clean_json(input_folder, temp_folder)
    json2csv.json_convert_csv(temp_folder, output_folder)

if __name__ == '__main__':
    data_folder = './data'
    data_folder = os.path.abspath(data_folder)
    cleaned_folder = './cleaned_data'
    cleaned_folder = os.path.abspath(cleaned_folder)
    cleaned_csv_folder = './cleaned_data_csv'
    cleaned_csv_folder = os.path.abspath(cleaned_csv_folder)
    traverse_and_clean_json2csv(data_folder, cleaned_folder, cleaned_csv_folder)
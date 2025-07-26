
import clean_json
import pandas as pd
import json
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

# 程序運行位置為: "~\questions-ai"

def json_convert_csv(input_folder, output_folder):

    # 遍歷文件夾並清洗 JSON 文件
    # clean_json.traverse_and_clean_json(input_folder, output_folder)

    # 假設你的 JSON 資料在 data.json 檔案
    # 收集所有 json 文件路徑
    json_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.json'):
                in_path = os.path.join(root, file)
                json_files.append(in_path)

    for in_path in tqdm(json_files, desc="轉換進度", unit="file"):
        with open(in_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        main_data = data['data']
        subject_list = main_data['subjectList']
        csv_file_path = os.path.join(output_folder, os.path.basename(in_path).replace('.json', '.csv'))
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        if isinstance(subject_list, list) and subject_list:
            df = pd.DataFrame(subject_list)
            df.to_csv(csv_file_path, index=False, encoding='utf-8')
            tqdm.write(f"已轉換: {in_path} -> {csv_file_path}")
        else:
            tqdm.write(f"跳過空 subjectList: {in_path}")


if __name__ == "__main__":
    input_folder = './cleaned_data/一級'
    input_folder = os.path.abspath(input_folder)
    output_folder = './cleaned_data_csv/一級'
    output_folder = os.path.abspath(output_folder)

    json_convert_csv(input_folder, output_folder)

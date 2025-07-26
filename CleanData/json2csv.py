import clean_json

import pandas as pd
import json
import os
from bs4 import BeautifulSoup

# 程序運行位置為: "~\questions-ai"

def json_convert_csv(input_folder, output_folder):

    # 遍歷文件夾並清洗 JSON 文件
    # clean_json.traverse_and_clean_json(input_folder, output_folder)

    # 假設你的 JSON 資料在 data.json 檔案
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            if file.endswith('.json'):
                in_path = os.path.join(root, file)
                with open(in_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                main_data = data['data']
                subject_list = main_data['subjectList']
                # 轉成 DataFrame
                df = pd.DataFrame(subject_list)

                # 輸出為 CSV 檔案
                csv_file_path = os.path.join(root, file.replace('.json', '.csv'))
                df.to_csv(csv_file_path, index=False, encoding='utf-8')
                print(f"已轉換: {in_path} -> {csv_file_path}")


if __name__ == "__main__":
    input_folder = '../cleaned_data/一級'
    output_folder = '../cleaned_data_csv/一級'

    # 假設你的 JSON 資料在 data.json 檔案
    with open(f'{input_folder}/Python_202103_1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    main_data = data['data']
    subject_list = main_data['subjectList']
    # 轉成 DataFrame
    df = pd.DataFrame(subject_list)

    # 輸出為 CSV 檔案
    df.to_csv(f'{output_folder}/Python_202103_1.csv', index=False, encoding='utf-8')
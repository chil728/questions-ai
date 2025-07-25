import pandas as pd
import json


if __name__ == "__main__":
    input_folder = '../data/一級'
    output_folder = '../cleaned_data/一級'

    # 假設你的 JSON 資料在 data.json 檔案
    with open(f'{input_folder}/Python_202103_1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    main_data = data['data']
    subject_list = main_data['subjectList']
    # 轉成 DataFrame
    df = pd.DataFrame(subject_list)

    # 輸出為 CSV 檔案
    df.to_csv(f'{output_folder}/Python_202103_1.csv', index=False, encoding='utf-8')
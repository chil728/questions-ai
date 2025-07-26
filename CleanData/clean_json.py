import json
import html
import os
from bs4 import BeautifulSoup

# 程序運行位置為: "~\questions-ai"

dirty_key = ["optionE", "optionF", "optionG"] #需清洗的數據
dirty_value = [None, '', 'null'] #需清洗的數據

# 遍歷文件夾並清洗 JSON 文件
def traverse_and_clean_json(folder, out_folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):
                in_path = os.path.join(root, file)
                # 保持目錄結構
                rel_path = os.path.relpath(in_path, folder)
                out_path = os.path.join(out_folder, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                clean_json_file(in_path, out_path)
                print(f"已處理: {in_path} -> {out_path}")

#清洗 HTML 格式殘留
def clean_html(raw_html, keep_img_src=False):
    if not isinstance(raw_html, str):
        return raw_html
    soup = BeautifulSoup(raw_html, 'html.parser')
    # 保留 <img> src 屬性（可選）
    if keep_img_src:
        for img in soup.find_all('img'):
            if img.has_attr('src'):
                return img['src']
    # 移除所有 HTML 標籤，保留文字
    text = soup.get_text(separator=' ', strip=True)
    # 處理 HTML 實體
    text = html.unescape(text)
    return text

# 清洗多餘數據
def clean_subject(subject, html_fields=None):
    html_fields = html_fields or [
        'title', 'optionA', 'optionB', 'optionC', 'optionD', 'optionE',
        'analyzeContent', 'analyzeVideo', 'comment'
    ]
    cleaned = {}
    for k, v in subject.items():
        if k in dirty_key or v in dirty_value:
            continue
        if k in html_fields and v:
            v = clean_html(v)
        if isinstance(v, str):
            v = zh_to_en_punct(v)
        cleaned[k] = v
    return cleaned

# 获取下一个可用的文件名
def get_next_filename(filename):
    """如果文件已存在，自動生成 filename_2, filename_3..."""
    base, ext = os.path.splitext(filename)
    i = 2
    new_filename = f"{base}_{i}{ext}"
    while os.path.exists(new_filename):
        i += 1
        new_filename = f"{base}_{i}{ext}"
    return new_filename

# 清洗數據的主函數
def clean_json_file(filename, out_filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    data = raw['data']
    cleaned_subjects = [clean_subject(subj) for subj in data['subjectList']]
    data['subjectList'] = cleaned_subjects
    cleaned = {'data': data, 'code': raw.get('code'), 'msg': raw.get('msg')}
    
    # Output cleaned data to file
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    target_file = out_filename
    if os.path.exists(out_filename):
        target_file = get_next_filename(out_filename)
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)



def zh_to_en_punct(text):
    with open('./cleanData/zh_en_punct.json', 'r', encoding='utf-8') as f:
        zh_en_dict = json.load(f)
    for zh, en in zh_en_dict.items():
        text = text.replace(zh, en)
    return text

# 讀取文件並轉換
def convert_file(infile, outfile):
    with open(infile, 'r', encoding='utf-8') as f:
        text = f.read()
    text = zh_to_en_punct(text)
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(text)

# 用法示例
# convert_file('source.txt', 'converted.txt')


if __name__ == '__main__':
    input_folder = './data/一級'
    input_folder = os.path.abspath(input_folder)
    output_folder = './cleaned_data/一級'
    output_folder = os.path.abspath(output_folder)
    traverse_and_clean_json(input_folder, output_folder)
    print("所有 JSON 文件已清洗完成。")
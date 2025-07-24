import json
import re
import html
from bs4 import BeautifulSoup

def remove_html_tags(text):
    """去除HTML标签并保留换行符"""
    if text is None:
        return ""
    # 使用BeautifulSoup保留换行符
    soup = BeautifulSoup(text, 'html.parser')
    # 处理特殊标签如表格
    for tag in soup.find_all(['table', 'tr', 'td']):
        tag.unwrap()
    # 保留换行符
    return soup.get_text().strip()

def transform_question(q):
    """转换单个题目为指定格式"""
    # 确定题型
    if q.get('type') == 1:
        question_type = "单选题"
    else:
        # 根据内容判断题型（示例中没有其他题型）
        if "判断" in q.get('title', '') or "对吗" in q.get('title', ''):
            question_type = "是非题"
        elif "编写" in q.get('title', '') or "代码" in q.get('title', ''):
            question_type = "编程题"
        else:
            question_type = "单选题"

    # 提取知识点
    knowledge = q.get('knowledgeNames', '').split(',') if q.get('knowledgeNames') else ["通用"]
    
    # 构建基础结构
    result = {
        "instruction": f"生成一道关于{'/'.join(knowledge)}的{question_type}",
        "input": {
            "题型": question_type,
            "知识点": knowledge,
            "语言": "Python",
            "难度": f"{q.get('difficultyLevel', '2')}級"
        },
        "output": {}
    }
    
    # 题目内容处理
    title = remove_html_tags(q.get('title', ''))
    
    # 根据题型构建output
    if question_type == "单选题":
        options = {}
        for opt in ['A', 'B', 'C', 'D', 'E']:
            opt_content = q.get(f'option{opt}')
            if opt_content and remove_html_tags(opt_content).strip():
                options[opt] = remove_html_tags(opt_content)
        
        result["output"] = {
            "题目": title,
            "选项": options,
            "答案": q.get('answer', ''),
            "解析": remove_html_tags(q.get('analyzeContent', ''))
        }
    
    elif question_type == "是非题":
        result["output"] = {
            "题目": title,
            "答案": "对" if q.get('answer') == "A" else "错",  # 根据选项转换
            "解析": remove_html_tags(q.get('analyzeContent', ''))
        }
    
    elif question_type == "编程题":
        result["output"] = {
            "题目": title,
            "要求": "按要求编写代码",
            "参考代码": "（根据题目要求实现代码）"
        }
    
    return result

def process_file(file_path, output_path):
    """处理单个JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    transformed = []
    for q in data['data']['subjectList']:
        try:
            transformed.append(transform_question(q))
        except Exception as e:
            print(f"处理题目出错: {q.get('id')}, 错误: {str(e)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

# 处理所有文件
file_pairs = [
    ('Python_202503_3.json', 'cleaned_202503_3.json')
]

for input_file, output_file in file_pairs:
    process_file(input_file, output_file)
    print(f"已处理: {input_file} -> {output_file}")

print("所有文件处理完成!")
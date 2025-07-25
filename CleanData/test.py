import json

json_str = '{"html": "<p class=\"MsoNormal\">內容</p>"}'
data = json.loads(json_str)  # 解析成字典

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
import os
path = 'data/mola-d/all'
text_name_list = os.listdir(path+'/_text/')
text_name_list.sort()
for text_name in text_name_list:
    text_path = os.path.join(path, '_text', text_name)
    with open(text_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
    text = ' '.join(lines) + '\n'
    pass

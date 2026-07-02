import json
import os

def json_to_txt(json_file_path, output_dir="output_txt"):
    # 创建输出目录，不存在则新建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取事件列表数组
    event_list = data.get("event_summary", [])
    if not event_list:
        print("未找到 event_summary 数据，程序退出")
        return

    # 遍历每一条事件对象
    for item in event_list:
        item_id = item.get("ID")
        if not item_id:
            print("存在缺失ID的数据，跳过本条")
            continue
        
        # 组装文本内容：排除ID字段，其余键值对分行
        content_lines = []
        for key, value in item.items():
            if key == "ID":
                continue
            line = f"{value}"
            content_lines.append(line)
        file_content = "\n".join(content_lines)

        # 生成txt文件路径
        txt_path = os.path.join(output_dir, f"{item_id}.txt")
        # 写入文件
        with open(txt_path, "w", encoding="utf-8") as out_f:
            out_f.write(file_content)
        print(f"已生成文件：{txt_path}")

if __name__ == "__main__":
    # ========== 修改此处配置 ==========
    # 你的json文件路径，相对/绝对路径均可
    JSON_FILE = "/Users/miami/Documents/project/rag-qa-system/data/eval_docs/scripts/split_merged.json"
    # 输出txt存放文件夹
    OUT_FOLDER = "/Users/miami/Documents/project/rag-qa-system/data/eval_docs/docs"
    # ==================================
    json_to_txt(JSON_FILE, OUT_FOLDER)

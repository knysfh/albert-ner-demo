import json
import re
from itertools import combinations

import ahocorasick as ah

DATA_DICT = {"O": 0, "B_CE": 1, "B_CM": 2, "B_EM": 3, "I_CE": 4, "I_CM": 5, "I_EM": 6}

train_file = "../data/train_data.json"
train_label_file = "../data/train_data_with_label.txt"
vaild_file = "../data/vaild_data.json"
vaild_label_file = "../data/vaild_data_with_label.txt"
test_file = "../data/test_data.json"
test_label_file = "../data/test_data_with_label.txt"


def text_process(input_file: str, out_file: str, max_length: int) -> None:
    """
    处理训练数据并生成带有类别标签的目标文件
    """
    with open(out_file, "a", encoding="utf-8") as wf:
        with open(input_file, "r", encoding="utf-8") as rf:
            while line := rf.readline():
                train_data_res = []
                train_data_json = json.loads(line)
                # 文本内容超过max_length则进行截断
                text = train_data_json["text"]
                text = re.sub(r"\n|\t|\r", "", text)
                if len(text) > max_length:
                    train_data_res.extend(cutdown_text(text=text, cut_len=max_length))
                else:
                    train_data_res.append(text)
                cause_effect_mention = train_data_json["cause_effect_mention"]
                cause_effect_list = train_data_json["cause_effect_list"]
                ac_machine = ah.Automaton()
                ac_machine.add_word(
                    cause_effect_mention, ("I_CE", len(cause_effect_mention))
                )
                for cause_effect in cause_effect_list:
                    cause_mention = cause_effect["cause_mention"]
                    effect_mention = cause_effect["effect_mention"]
                    ac_machine.add_word(cause_mention, ("I_CM", len(cause_mention)))
                    ac_machine.add_word(effect_mention, ("I_EM", len(effect_mention)))
                ac_machine.make_automaton()
                for train_data in train_data_res:
                    label_data = ["1" + "0" * (len(DATA_DICT) - 1)] * len(train_data)
                    for item in ac_machine.iter(train_data):
                        start_pos = item[0] - item[1][1] + 1
                        end_pos = item[0] + 1
                        for num in range(start_pos, end_pos):
                            class_list = list(label_data[num])
                            class_list[0] = "0"
                            class_name = item[1][0]
                            if num == start_pos:
                                class_name = class_name.replace("I_", "B_")
                            class_list[DATA_DICT[class_name]] = "1"
                            label_data[num] = "".join(class_list)
                    for i in range(len(train_data)):
                        wf.write(train_data[i] + "\t" + label_data[i] + "\n")
                    wf.write("\n")


def cutdown_text(text: str, cut_len: int) -> list:
    """
    将文本内容超过字数限制的文本截断
    """
    text_len = len(text)
    cur_len = 0
    text_res_list = []
    while cur_len < text_len:
        if cur_len + cut_len >= text_len:
            text_res_list.append(text[cur_len : cur_len + cut_len])
            cur_len += cur_len
            break
        tmp_text = text[cur_len : cur_len + cut_len]
        re_res = re.search("(.*。)*", tmp_text)
        if not re_res.group():
            re_res = re.search(".*(。|”|,|，|：|:|？|\?)+", tmp_text)
        text_res = (
            text[cur_len : re_res.span()[1] + cur_len]
            if re_res
            else text[cur_len : cur_len + cut_len]
        )
        text_res_list.append(text_res)
        cur_len += len(text_res)
    return text_res_list


def label_combination(data_list: list, n: int) -> list:
    """获取所有标签的组合"""
    result_list = []
    for combin in combinations(data_list, n):
        tmp_set = set(combin)
        if (
            # 过滤标签组合内存在互斥标签的数据
            tmp_set.issuperset({"B_CE", "I_CE"})
            or tmp_set.issuperset({"B_CM", "I_CM"})
            or tmp_set.issuperset({"B_EM", "I_EM"})
        ):
            continue
        result_list.append(list(combin))
    return result_list


def creat_label_file(out_file: str) -> None:
    """
    按照'A Multilabel Approach to Portuguese Clinical Named Entity Recognition'论文的方法构建新标签,
    将'O, B_CE, B_CM, B_EM, I_CE, I_CM, I_EM'标签转换为one hot形式
    单标签形式:O标签 -> 1 0 0 0 0 0 0
    多标签形式:B_CE, I_CM, I_EM标签 -> 0 1 0 0 0 1 1
    """
    data_key = list(DATA_DICT.keys())
    data_key.remove("O")
    new_class_label = "0" * len(DATA_DICT)
    label_dict = dict()
    label_index = 1
    for num in range(1, (int(len(data_key) / 2) + 1)):
        for combin in label_combination(data_key, num):
            class_list = list(new_class_label)
            for label in combin:
                class_list[DATA_DICT[label]] = "1"
            label_dict["".join(class_list)] = [combin, label_index]
            label_index += 1
    # 添加O标签
    class_list = list(new_class_label)
    class_list[DATA_DICT["O"]] = "1"
    label_dict["".join(class_list)] = [["O"], 0]
    with open(out_file, "a") as wf:
        json.dump(label_dict, wf)


def get_data_dict(file_path: str) -> dict:
    """加载标签数据"""
    with open(file_path, "r") as rf:
        data_dict = json.load(rf)
    return data_dict


if __name__ == "__main__":
    text_process(input_file=train_file, out_file=train_label_file, max_length=127)
    text_process(input_file=vaild_file, out_file=vaild_label_file, max_length=127)
    text_process(input_file=test_file, out_file=test_label_file, max_length=127)
    creat_label_file("../data/label_dict.txt")

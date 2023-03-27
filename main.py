import math
import os
from typing import Tuple

import bert
import numpy as np
import tensorflow as tf
from bert.tokenization.albert_tokenization import FullTokenizer
from tensorflow import keras

from src.eval import Metrics
from src.utils import get_data_dict

max_seq_length = 128
batch_size = 32
epoch_num = 50
model_dir = "./albert_base"
model_ckpt = os.path.join(model_dir, "albert_model.ckpt")
vocab_file = os.path.join(model_dir, "vocab.txt")
label_dict = get_data_dict("data/label_dict.txt")
train_file_path = "data/train_data_with_label.txt"
vaild_file_path = "data/vaild_data_with_label.txt"
test_file_path = "data/test_data_with_label.txt"


def get_masks(tokens: list, max_seq_length: int) -> list:
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens: list, max_seq_length: int) -> list:
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens: list, tokenizer: FullTokenizer, max_seq_length: int) -> list:
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def get_dataset(file_path: str, label_dict: dict) -> tuple[list, list]:
    """Get train/vaild/test dataset from file"""
    sequence = []
    label = []
    with open(file_path, "r") as rf:
        sub_seq = ""
        sub_label = []
        while line := rf.readline():
            if line != "\n":
                line_data = line.rstrip().split("\t")
                sub_seq += line_data[0]
                sub_label.append(label_dict[line_data[1]][1])
            else:
                sequence.append(sub_seq)
                label.append(sub_label)
                sub_seq = ""
                sub_label = []
    return (sequence, label)


def map_dataset_to_dict(input_ids, token_type_ids, label):
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
    }, label


def encode_dataset(
    dataset: Tuple[list, list], tokenizer: FullTokenizer
) -> tf.data.Dataset:
    """
    编码数据集
    """
    input_data, input_label = dataset
    input_ids_list = []
    token_type_ids_list = []
    for i in range(len(input_data)):
        data_token = tokenizer.tokenize(input_data[i])
        data_token = ["[CLS]"] + data_token
        input_ids_list.append(get_ids(data_token, tokenizer, max_seq_length))
        token_type_ids_list.append(get_segments(data_token, max_seq_length))
        # 为label添加pad和[CLS]的标签
        input_label[i].insert(0, 0)
        input_label[i].extend([0] * (max_seq_length - len(input_label[i])))
    # 将label转为one hot形式
    end_label = tf.keras.utils.to_categorical(input_label)

    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, token_type_ids_list, end_label)
    ).map(map_dataset_to_dict)


def create_learning_rate_scheduler(
    max_learn_rate=1e-5,
    end_learn_rate=1e-7,
    warmup_epoch_count=20,
    total_epoch_count=90,
):
    """
    动态调整学习率
    """

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate * math.exp(
                math.log(end_learn_rate / max_learn_rate)
                * (epoch - warmup_epoch_count + 1)
                / (total_epoch_count - warmup_epoch_count + 1)
            )
        return float(res)

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lr_scheduler, verbose=1
    )

    return learning_rate_scheduler


def creat_model() -> tf.keras.Model:
    # 构建模型
    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, name="albert")

    input_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype="int32", name="input_ids"
    )
    token_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype="int32", name="token_type_ids"
    )
    output = l_bert([input_ids, token_type_ids])
    logits = tf.keras.layers.Dense(units=768, activation="tanh")(output)
    logits = tf.keras.layers.Dropout(0.5)(logits)
    y_pred = tf.keras.layers.Dense(units=len(label_dict), activation="softmax")(logits)

    model = tf.keras.Model(inputs=[input_ids, token_type_ids], outputs=y_pred)

    return model


def eval_model():
    """
    评估模型
    """
    # id -> label的字典例如 {0: "O", 1: "B_CM"...}
    id2label_dict = dict()
    for val in label_dict.values():
        id2label_dict[val[1]] = ",".join(val[0])

    tokenizer = bert.albert_tokenization.FullTokenizer(vocab_file)
    init_test_data = get_dataset(test_file_path, label_dict)
    test_data = encode_dataset(init_test_data, tokenizer)
    test_data = test_data.batch(batch_size=batch_size)
    # 加载模型权重
    model = creat_model()
    model.load_weights("ccks-ner.h5")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    model.summary()
    pred_label = []
    real_label = []
    pred_res = model.predict(test_data)
    for seq_val in pred_res:
        pred_label.append([id2label_dict[np.argmax(i)] for i in seq_val])
    for t_label in init_test_data[1]:
        real_label.append([id2label_dict[i] for i in t_label])
    eval_res = Metrics(real_label, pred_label, remove_O=True)
    eval_res.report_scores()


def train_model():
    """
    训练模型
    """
    tokenizer = bert.albert_tokenization.FullTokenizer(vocab_file)
    init_train_data = get_dataset(train_file_path, label_dict)
    init_vaild_data = get_dataset(vaild_file_path, label_dict)
    train_data = encode_dataset(init_train_data, tokenizer)
    train_data = train_data.shuffle(buffer_size=5000, seed=3407).batch(
        batch_size=batch_size
    )
    vaild_data = encode_dataset(init_vaild_data, tokenizer)
    vaild_data = vaild_data.batch(batch_size=batch_size)
    model = creat_model()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )

    model.summary()

    model.fit(
        x=train_data,
        epochs=epoch_num,
        validation_data=vaild_data,
        callbacks=[
            create_learning_rate_scheduler(total_epoch_count=epoch_num),
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ],
    )

    model.save_weights("./ccks-ner.h5")


if __name__ == "__main__":
    train_model()
    eval_model()

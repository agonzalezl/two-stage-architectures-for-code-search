
import gzip
import json
import os
from pathlib import Path
import ntpath
import numpy as np

from tqdm import tqdm
import tensorflow as tf
import transformers
import random
import pathlib

class TFRecordParser():

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    number_of_tokens = tokenizer.vocab_size

    def __init__(self):
        pass

    @staticmethod
    def to_tfrecord_features(tokenized_doc, tokenized_code, tokenized_negative, similarity):

        # int_list1 = tf.train.Int64List(value = [data_record['int_data']])
        _tokenized_doc = tf.train.Int64List(value=tokenized_doc.flatten())
        _tokenized_code = tf.train.Int64List(value=tokenized_code.flatten())
        _tokenized_negative = tf.train.Int64List(value=tokenized_negative.flatten())

        _similarity = tf.train.FloatList(value=[similarity])

        feature_key_value_pair = {
            'tokenized_doc': tf.train.Feature(int64_list=_tokenized_doc),
            'tokenized_code': tf.train.Feature(int64_list=_tokenized_code),
            'tokenized_negative': tf.train.Feature(int64_list=_tokenized_negative),
            'similarity': tf.train.Feature(float_list=_similarity)
        }

        features = tf.train.Features(feature=feature_key_value_pair)

        # Create Example object with features
        return tf.train.Example(features=features)

    @staticmethod
    def from_jsonlgz_to_tfrecord(path, target_path):
        # Example: python, train, 0
        language, dataset, chunk = ntpath.basename(path).split(".")[0].split("_")[0:3]
        with gzip.open(path, 'r') as pf:
            data = pf.readlines()

        data_shuffled = data.copy()
        random.shuffle(data_shuffled)

        with tf.io.TFRecordWriter(
                target_path + language + "/" + dataset + "/" + str(chunk) + ".tfrecord") as tfwriter:
            for d in data:
                line_a = json.loads(str(d, encoding='utf-8'))

                docstring_tokens = " ".join(line_a["docstring_tokens"])
                code_tokens = " ".join(line_a["code_tokens"])

                tokenized_doc = TFRecordParser.tokenize(docstring_tokens)
                tokenized_code = TFRecordParser.tokenize(code_tokens)

                # Negative sampling
                ramdom_example = data_shuffled.pop(0)
                ramdom_example = json.loads(str(ramdom_example, encoding='utf-8'))
                negative_description = " ".join(ramdom_example["docstring_tokens"])
                tokenized_negative = TFRecordParser.tokenize(negative_description)

                # Create Example object with features
                example = TFRecordParser.to_tfrecord_features(tokenized_doc, tokenized_code, tokenized_negative, 0.0)

                tfwriter.write(example.SerializeToString())

    @staticmethod
    def tokenize(string):
        encoded = TFRecordParser.tokenizer.batch_encode_plus(
            [string],
            add_special_tokens=True,
            max_length=45,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors="np"
        )["input_ids"][0]
        return encoded  # encoded    encoded["input_ids"][0]


    @staticmethod
    def generate_tfrecords(files, target_path):
        pbar = tqdm(total=len(files))
        for file in files:
            TFRecordParser.from_jsonlgz_to_tfrecord(file, target_path)
            pbar.update(1)
        pbar.close()

    @staticmethod
    def extract_fn(data_record):
        features = {

            'tokenized_doc': tf.io.FixedLenFeature((45), tf.int64),
            # tf.io.FixedLenFeature([], tf.int64), #tf.io.VarLenFeature(tf.int64),
            'tokenized_code': tf.io.FixedLenFeature((45), tf.int64),
            'tokenized_negative': tf.io.FixedLenFeature((45), tf.int64),
            'similarity': tf.io.FixedLenFeature([], tf.float32)

        }
        sample = tf.io.parse_single_example(data_record, features)
        return (sample["tokenized_code"], sample["tokenized_doc"], sample["tokenized_negative"]), sample[
            "similarity"]  # sample["tokenized_doc"], sample["tokenized_negative"]

    @staticmethod
    def generate_dataset(tfr_files, batch_size = 32):
        # Initialize all tfrecord paths
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed

        dataset = tf.data.TFRecordDataset(tfr_files)

        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        dataset = dataset.map(TFRecordParser.extract_fn)
        dataset = dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        dataset = dataset.batch(batch_size)
        return dataset


if __name__ == "__main__":
    script_path = str(pathlib.Path(__file__).parent)+"/"

    python_files = sorted(Path(script_path+'../data/codesearchnet/python/final/jsonl/').glob('**/*.gz'))
    go_files = sorted(Path(script_path+'../data/codesearchnet/go/final/jsonl/train/').glob('**/*.gz'))
    java_files = sorted(Path(script_path+'../data/codesearchnet/java/final/jsonl/train/').glob('**/*.gz'))
    php_files = sorted(Path(script_path+'../data/codesearchnet/php/final/jsonl/train/').glob('**/*.gz'))
    javascript_files = sorted(Path(script_path+'../data/codesearchnet/javascript/final/jsonl/train/').glob('**/*.gz'))
    ruby_files = sorted(Path(script_path+'../data/codesearchnet/ruby/final/jsonl/train/').glob('**/*.gz'))

    all_files = python_files  # + go_files + java_files + php_files + javascript_files + ruby_files

    languages = ["python", "java", "go", "javascript"]
    target_path = script_path+"../data/codesearchnet/tfrecord/"

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for language in languages:
        if not os.path.exists(target_path + language):
            os.makedirs(target_path + language)
            os.makedirs(target_path + language + "/train/")
            os.makedirs(target_path + language + "/valid/")
            os.makedirs(target_path + language + "/test/")

    TFRecordParser.generate_tfrecords(all_files, target_path)

    print(f'Total number of files: {len(all_files):,}')
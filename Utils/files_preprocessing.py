import os
from pathlib import Path

from TextPreprocessing import TextPreprocess


def preprocess_all_folders():
    dataset_path = os.path.join(os.path.join(os.path.join(Path(os.getcwd()), 'Data'), 'Datasets'))
    dataset_dir = os.path.join(dataset_path, 'aclImdb')

    dataset_train_dir = os.path.join(dataset_dir, 'train')

    dataset_train_dir_pos = os.path.join(dataset_train_dir, 'pos')
    files = os.listdir(dataset_train_dir_pos)
    preprocess_all_files(dataset_train_dir_pos, files)

    dataset_train_dir_neg = os.path.join(dataset_train_dir, 'neg')
    files = os.listdir(dataset_train_dir_neg)
    preprocess_all_files(dataset_train_dir_neg, files)

    dataset_test_dir = os.path.join(dataset_dir, 'test')

    dataset_test_dir_pos = os.path.join(dataset_test_dir, 'pos')
    files = os.listdir(dataset_test_dir_pos)
    preprocess_all_files(dataset_test_dir_pos, files)

    dataset_test_dir_neg = os.path.join(dataset_test_dir, 'neg')
    files = os.listdir(dataset_test_dir_neg)
    preprocess_all_files(dataset_test_dir_neg, files)


def preprocess_all_files(path, files):
    for file in files:
        newfile = os.path.join(path, file)
        replaceAll(newfile)


def replaceAll(file_name):
    with open(file_name, 'r', encoding="utf-8") as file:
        text = file.read()  # read file into memory

    text = TextPreprocess.preprocess(text)  # make replacements

    with open(file_name, 'w', encoding="utf-8") as file:
        file.write(text)  # rewrite the file

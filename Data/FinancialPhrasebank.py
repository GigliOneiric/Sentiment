import os
import pandas as pd
from datasets import load_dataset, DownloadMode


def write_txt(dataset, train_dir, test_dir):
    i = 0

    for row in dataset.values:
        if i % 2 == 1:
            filename = os.path.join(train_dir, 'FP' + str(i) + '.txt')
            f = open(filename, 'w', encoding='utf-8')
            f.write(row[0])
            f.close()
            i += 1
        elif i % 2 == 0:
            filename = os.path.join(test_dir, 'FP' + str(i) + '.txt')
            f = open(filename, 'w', encoding='utf-8')
            f.write(row[0])
            f.close()
            i += 1


class FinancialPhrasebank:

    def __init__(self, dataset_path, dataset_dir, train_dir, test_dir, train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir):
        self.pos_df = None
        self.neg_df = None
        self.dataset_path = dataset_path
        self.dataset_file = None
        self.dataset = None
        self.dataset_dir = dataset_dir
        self.train_dir = train_dir
        self.train_pos_dir = train_pos_dir
        self.train_neg_dir = train_neg_dir
        self.test_dir = test_dir
        self.test_pos_dir = test_pos_dir
        self.test_neg_dir = test_neg_dir

        self.load_datasets()

    def load_datasets(self):

        """
        ## Set paths to the directory's
        """

        self.dataset_file = os.path.join(self.dataset_path, 'financial_phrasebank.csv')

        """
        ## Load the data: IMDB movie review sentiment classification.
        """

        if not os.path.exists(self.dataset_file):
            self.dataset = load_dataset(path='financial_phrasebank',
                                        name='sentences_allagree',
                                        download_mode=DownloadMode.FORCE_REDOWNLOAD)

            for split, data in self.dataset.items():
                data.to_csv(self.dataset_file, index=None)

            self.write_to_folder()

    def write_to_folder(self):
        self.dataset = pd.read_csv(self.dataset_file)

        self.pos_df = self.dataset[self.dataset['label'] == 2]
        write_txt(self.pos_df, self.train_pos_dir, self.test_pos_dir)

        self.neg_df = self.dataset[self.dataset['label'] == 0]
        write_txt(self.neg_df, self.train_neg_dir, self.test_neg_dir)

from torch.utils import data
import numpy as np
import utils.data_reader as data_reader
import utils.glove_pruner as glove_pruner
from tqdm import tqdm
import torch
from torch.autograd import Variable

DATA_PATH = "data/" #change to "../data" if want to run locally
TRAINING_SET_FILE = DATA_PATH + "beer-ratings/train.csv"
WORD_TO_EMBEDDING_FILE = DATA_PATH + "glove_prunned.txt"

REVIEWS_FIELD = "review/text"

TRAIN_SET = data_reader.read_data(TRAINING_SET_FILE)
WORD_TO_EMBEDDING = glove_pruner.load_words(WORD_TO_EMBEDDING_FILE)


class Database(object):
    TOKENS_FIELD = "processed/tokens"
    EMBEDDINGS_FIELD = "processed/embeddings"
    def __init__(self):
        raw_train_set_length = len(TRAIN_SET[REVIEWS_FIELD])
        train_set_end = int(0.7 * raw_train_set_length)
        dev_set_end = int(0.9 * raw_train_set_length)

        self.train_set = BeerReviewsDataset(TRAIN_SET, start=0, end=train_set_end)
        self.dev_set = BeerReviewsDataset(TRAIN_SET, start=train_set_end, end=dev_set_end)
        self.test_set = BeerReviewsDataset(TRAIN_SET, start=dev_set_end, end=raw_train_set_length)



class BeerReviewsDataset(data.Dataset):
    MAX_LENGTH = 200
    EMBEDDING_DIMS = 300
    def __init__(self, datasets, start, end):
        self.start = start
        self.end = end
        self.datasets = datasets

    def __len__(self):
        return (self.end - self.start) // 10

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError
        
        datapoint = {}

        for field in self.datasets:
            datapoint[field] = self.datasets[field][self.start + idx]
        
        text = datapoint[REVIEWS_FIELD]
        tokens, embeddings = self._process_text(text)
        datapoint[Database.TOKENS_FIELD] = tokens
        datapoint[Database.EMBEDDINGS_FIELD] = embeddings

        return datapoint
    
    def _process_text(self, text):
        tokens = list(filter(lambda word: word in WORD_TO_EMBEDDING, data_reader.tokenize(text)))
        embeddings_list = list(map(lambda word: WORD_TO_EMBEDDING[word], tokens))
        embeddings_np_array = np.zeros((self.MAX_LENGTH, self.EMBEDDING_DIMS))
        if len(embeddings_list) > 0:
            embeddings_np_array[:min(self.MAX_LENGTH, len(embeddings_list))] = np.array(embeddings_list[:self.MAX_LENGTH])
        return tokens, torch.from_numpy(embeddings_np_array)


if __name__ == "__main__":
    database = Database()

    print(len(database.train_set))
    print(len(database.dev_set))
    print(len(database.test_set))

    exit()

    print(database.test_set[0][Database.EMBEDDINGS_FIELD])

    import matplotlib.pyplot as plt
    plt.hist(map(float, database.test_set.datasets['review/overall']))
    plt.show()

    """
    lengths = []
    for i in tqdm(xrange(len(database.train_set))):
        new_length = len(database.train_set[i][TOKENS_FIELD])
        lengths.append(new_length)
    

    print(len(filter(lambda l: l>180, lengths)))
    """

    """
    import matplotlib.pyplot as plt

    plt.hist(lengths)
    plt.show()
    """





    




    


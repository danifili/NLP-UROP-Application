import utils.data_reader
from tqdm import tqdm

def load_words(glove_filename):
    word_to_vec = {}
    with open(glove_filename) as glove_file:
        for line in glove_file:
            split_line = line.split()
            word, vec = str(split_line[0].lower()), list(map(float, split_line[1:]))
            word_to_vec[word] = vec
    return word_to_vec

def get_dataset_words(filename_list, field):
    words = set()
    for filename in tqdm(filename_list):
        dataset = data_reader.read_data(filename)
        for text in dataset[field]:
            words.update(data_reader.tokenize(text, None))
    return words

def pruner(words, word_to_vec, outfilename):
    with open(outfilename, 'w') as outfile:
        for word in tqdm(words):
            if word in word_to_vec:
                outfile.write(str(word) + " " + " ".join(map(str, word_to_vec[word])) + "\n")


if __name__ == "__main__":
    data_words = get_dataset_words(['beer-ratings/train.csv', 'beer-ratings/test.csv'], 'review/text')
    glove_word_to_vec = load_words('glove.42B.300d.txt')
    pruner(data_words, glove_word_to_vec, 'glove_prunned.txt')
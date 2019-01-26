from sklearn.feature_extraction.text import CountVectorizer
import csv

def read_data(filename):
    dataset = []
    csv_file = open(filename, "r")
    reader = csv.reader(csv_file)
    for line in reader:
        if len(dataset) == 0:
            for field in line:
                dataset.append({"field": field, "data": []})
        else:
            for i, data_point in enumerate(line):
                dataset[i]["data"].append(data_point)
    
    return {data_field["field"]: data_field["data"] for data_field in dataset}

def tokenize(text, stop_words = 'english'):
    vectorizer = CountVectorizer(stop_words = stop_words)
    analyser = vectorizer.build_analyzer()
    return analyser(text)  

if __name__ == "__main__":
    train_file = "beer-ratings/train.csv"
    test_file = "beer-ratings/test.csv"

    dataset = read_data(train_file)

    for field in dataset:
        if field == "review/text":
            print (tokenize(dataset[field][1], None))
            break
        

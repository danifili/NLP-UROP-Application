from models.ffnn import FFNN
from models.cnn import CNN
from models.gru import GRU
from utils.database import Database
from torch.utils import data
import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import pickle

model = 'gru'
field = 'palate'
REVIEW_FIELD = "review/" + field

def train_epoch(batch, encoder, classifier, optimizer_encoder, optimizer_classifier):
    criterion = torch.nn.MSELoss()

    encoded_sentences = encoder(Variable(batch[Database.EMBEDDINGS_FIELD]).float())
    out = classifier(encoded_sentences)

    loss = criterion(out, Variable(batch_reviews_to_torch(batch[REVIEW_FIELD])).float())

    optimizer_encoder.zero_grad()
    loss.backward(retain_graph=True)
    optimizer_encoder.step()

    optimizer_classifier.zero_grad()
    loss.backward()
    optimizer_classifier.step()

def review_to_torch(review):
    int_value = int(round(float(review) * 2)) - 2
    return [1 if int_value > i else 0 for i in range(8)]

def batch_reviews_to_torch(reviews):
    return torch.LongTensor(list(map(review_to_torch, reviews)))

def torch_to_review(torch_repr, tau=0.5):
    for i in range(len(torch_repr)):
        if torch_repr[i] < tau:
            return 1 + 0.5 * i
    return 5.0

def predict(encoder, classifier, dataset):
    dataloader = data.DataLoader(dataset, batch_size=100)
    predictions = []
    results = []
    for batch in tqdm(dataloader):
        encoded_sentences = encoder(Variable(batch[Database.EMBEDDINGS_FIELD]).float())
        out = classifier(encoded_sentences)
        for torch_review in out.data:
            predictions.append(torch_to_review(torch_review))
        results.extend(batch[REVIEW_FIELD])
    
    return np.array(predictions), np.array(list(map(float, results)))
    
if __name__ == "__main__":
    print (model, field)

    database = Database()
    batch_size = 40
    dataloader = data.DataLoader(database.train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    if model == "cnn":
        encoder = CNN(300, 200, 3)
    else:
        encoder = GRU(300, 200)
    classifier = FFNN(200, 50, 8)

    learning_rate = 1e-3
    n_epochs = 10

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=learning_rate)


    train_errors = []
    dev_errors = []
    test_errors = []

    for i in range(n_epochs):
        for batch in tqdm(dataloader):
            train_epoch(batch, encoder, classifier, optimizer_encoder, optimizer_classifier)

        train_pred, train_ans = predict(encoder, classifier, database.train_set)
        train_err = np.mean((train_pred-train_ans) ** 2)
        print (train_err)
        train_errors.append(train_err)

        dev_pred, dev_ans = predict(encoder, classifier, database.dev_set)
        dev_err = np.mean((dev_pred-dev_ans) ** 2)
        print (dev_err)
        print (dev_pred[:50], dev_ans[:50])
        dev_errors.append(dev_err)

        test_pred, test_ans = predict(encoder, classifier, database.test_set) 
        test_err = np.mean((test_pred-test_ans) ** 2)   
        print (test_err)
        test_errors.append(test_err)
    
    print (train_errors)
    print (dev_errors)
    print (test_errors)

    with open(model + '/' + field + '/' + 'predictions_train.pickle', 'wb') as handle:
        pickle.dump(train_pred, handle)
    
    with open(model + '/' + field + '/' + 'answers_train.pickle', 'wb') as handle:
        pickle.dump(train_ans, handle)    
    
    with open(model + '/' + field + '/' +'predictions_dev.pickle', 'wb') as handle:
        pickle.dump(dev_pred, handle)
    
    with open(model + '/' + field + '/' +'answers_dev.pickle', 'wb') as handle:
        pickle.dump(dev_ans, handle)
    
    with open(model + '/' + field + '/' +'predictions_test.pickle', 'wb') as handle:
        pickle.dump(test_pred, handle)
    
    with open(model + '/' + field + '/' +'answers_test.pickle', 'wb') as handle:
        pickle.dump(test_ans, handle)

    with open(model + '/' + field + '/' +'train_errors.pickle', 'wb') as handle:
        pickle.dump(train_errors, handle)

    with open(model + '/' + field + '/' +'dev_errors.pickle', 'wb') as handle:
        pickle.dump(dev_errors, handle)
    
    with open(model + '/' + field + '/' +'test_errors.pickle', 'wb') as handle:
        pickle.dump(test_errors, handle)






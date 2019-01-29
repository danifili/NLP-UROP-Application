import pickle
import matplotlib.pyplot as plt



if __name__ == "__main__":
    model = 'cnn'
    field = 'taste'
    with open(model + '/' + field + 'predictions_train.pickle', 'rb') as handle:
        train_pred = pickle.load(handle)
    
    with open(model + '/' + field + 'answers_train.pickle', 'rb') as handle:
        train_ans = pickle.load(handle)    
    
    with open(model + '/' + field + 'predictions_dev.pickle', 'rb') as handle:
        dev_pred = pickle.load(handle)
    
    with open(model + '/' + field + 'answers_dev.pickle', 'rb') as handle:
        dev_ans = pickle.load(handle)  
    
    with open(model + '/' + field + 'predictions_test.pickle', 'rb') as handle:
        test_pred = pickle.load(handle)
    
    with open(model + '/' + field + 'answers_test.pickle', 'rb') as handle:
        test_ans = pickle.load(handle)

    for pred, ans in [(train_pred, train_ans), (dev_pred, dev_ans), (test_pred, test_ans)]:
        plt.hist(pred, bins=map(lambda x: x/2.0, range(1, 12)))
        plt.figure()
        plt.hist(ans, bins=map(lambda x: x/2.0, range(1, 12)))
        plt.figure()
    
    plt.show()
        
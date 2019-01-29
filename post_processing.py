import pickle
import matplotlib.pyplot as plt
model = 'cnn'
field = 'overall'
def plot_predictions_vs_answers():
    with open(model + '/' + field + '/' + 'predictions_train.pickle', 'rb') as handle:
        train_pred = pickle.load(handle)
    
    with open(model + '/' + field + '/' + 'answers_train.pickle', 'rb') as handle:
        train_ans = pickle.load(handle)    
    
    with open(model + '/' + field + '/' + 'predictions_dev.pickle', 'rb') as handle:
        dev_pred = pickle.load(handle)
    
    with open(model + '/' + field + '/' + 'answers_dev.pickle', 'rb') as handle:
        dev_ans = pickle.load(handle)  
    
    with open(model + '/' + field + '/' + 'predictions_test.pickle', 'rb') as handle:
        test_pred = pickle.load(handle)
    
    with open(model + '/' + field + '/' + 'answers_test.pickle', 'rb') as handle:
        test_ans = pickle.load(handle)

    for dataset_type, pred, ans in [('train', train_pred, train_ans), ('dev', dev_pred, dev_ans), ('test', test_pred, test_ans)]:
        plt.figure()
        plt.xlabel('ratings')
        plt.ylabel('quantities')
        plt.title(model + " " + field + " " + dataset_type + " set" + " predictions")
        plt.hist(pred, bins=map(lambda x: x/2.0, range(1, 12)), align='left', rwidth=0.5)
        plt.figure()
        plt.xlabel('ratings')
        plt.ylabel('quantities')
        plt.title(model + " " + field + " " + dataset_type + " set" + " answers")
        plt.hist(ans, bins=map(lambda x: x/2.0, range(1, 12)), align='left', color='orange', rwidth=0.5)

def plot_errors(train_errors, dev_errors, test_errors):
    plt.figure()
    plt.xlabel("number of epochs")
    plt.ylabel("mean square error")
    plt.title(model + " " + field + " errors")
    plt.plot(train_errors, label="train error")
    plt.plot(dev_errors, label="dev error")
    plt.plot(test_errors, label="test error")
    plt.legend()

if __name__ == "__main__":
    train = [0.31504761904761902, 0.29979047619047616, 0.28173333333333334, 0.28964761904761904, 0.30588571428571426, 0.26680952380952383, 0.25520952380952383, 0.2580095238095238, 0.26683809523809526, 0.24959999999999999]
    dev = [0.32173333333333332, 0.32269999999999999, 0.31613333333333332, 0.31863333333333332, 0.32963333333333333, 0.31240000000000001, 0.30603333333333332, 0.30980000000000002, 0.31790000000000002, 0.31030000000000002]
    test = [0.32519999999999999, 0.32626666666666665, 0.30740000000000001, 0.31846666666666668, 0.32566666666666666, 0.2994, 0.30033333333333334, 0.30846666666666667, 0.31, 0.31473333333333331]
    plot_predictions_vs_answers()
    #plot_errors(train, dev, test)
    plt.show()
        
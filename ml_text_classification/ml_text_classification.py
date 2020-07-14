import numpy as np
import csv
from scipy.sparse import csr_matrix
import string

MAX_ROW_NUM = 1000
CLASS_NUM = 5

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_lexicon(file_name):
    lexicon = {'<UNK>': 0}
    cnt = 0
    with open('data\\' + file_name, encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    continue
                word_list = tokenize(row[2])
                for word_idx, word in enumerate(word_list):
                    token1 = ()
                    token2 = ()
                    if word_idx == 0:
                        token1 = ('<START>', word)
                    if word_idx == len(word_list)-1:
                        token2 = (word, '<STOP>')
                    if len(token1) == 0:
                        token1 = (word_list[word_idx-1], word)
                    if len(token2) == 0:
                        token2 = (word, word_list[word_idx+1]) 

                    if word not in lexicon:
                        lexicon[word] = cnt
                        cnt += 1
                    if token1 not in lexicon:
                        lexicon[token1] = cnt
                        cnt += 1
                    if token2 not in lexicon:
                        lexicon[token2] = cnt
                        cnt += 1

    np.save('lexicon.npy', lexicon) 
    return lexicon

def n_gram_extractor(file_name):

    MARK = np.zeros(1)
    try:
        MARK[0] = np.loadtxt("m.txt", dtype=int)
    except:
        pass
    try:
        lexicon = np.load('lexicon.npy', allow_pickle=True).item()
    except:
        lexicon = get_lexicon(file_name)
    Y = np.zeros((MAX_ROW_NUM, 1))
    X = np.zeros((MAX_ROW_NUM, len(lexicon)))
    while(True):
        with open('data\\' + file_name, encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    continue
                if row_idx <= MARK[0]:
                    continue
                MARK[0] = row_idx
                word_list = tokenize(row[2])
                for word_idx, word in enumerate(word_list):
                    token1 = ()
                    token2 = ()
                    if word_idx == 0:
                        token1 = ('<START>', word)
                    if word_idx == len(word_list)-1:
                        token2 = (word, '<STOP>')
                    if len(token1) == 0:
                        token1 = (word_list[word_idx-1], word)
                    if len(token2) == 0:
                        token2 = (word, word_list[word_idx+1]) 

                    if word not in lexicon:
                        word = '<UNK>'
                    if token1 not in lexicon:
                        token1 = '<UNK>'
                    if token2 not in lexicon:
                        token2 = '<UNK>'

                    X[row_idx%MAX_ROW_NUM][lexicon[word]] = 1
                    X[row_idx%MAX_ROW_NUM][lexicon[token1]] = 1
                    X[row_idx%MAX_ROW_NUM][lexicon[token2]] = 1

                Y[row_idx%MAX_ROW_NUM][0] = int(row[3])
                if row_idx % MAX_ROW_NUM == 0:
                    np.savetxt("m.txt", MARK, fmt="%d")          
                    yield X, Y
                    X = np.zeros((MAX_ROW_NUM, len(lexicon)))
                    Y = np.zeros((MAX_ROW_NUM, 1))
            MARK = np.zeros(1)
            np.savetxt("m.txt", MARK, fmt="%d")
            yield X, Y

def sigmoid(z):
        return 1 / (1+np.exp(-z))

class LogisticPredictor(object):

    def __init__(self, train_file):
        self.generator = n_gram_extractor('train.tsv')
        self.X_train, self.Y_train = next(self.generator)
        self.num_train = self.X_train.shape[0]
        self.num_feature = self.X_train.shape[1]
        self.w = np.zeros((CLASS_NUM, self.num_feature, 1))
        self.b = np.zeros(CLASS_NUM)

    def read_new_train(self):
        self.X_train, self.Y_train = next(self.generator)
        self.num_train = self.X_train.shape[0]
        self.num_feature = self.X_train.shape[1]
        while(self.w.shape[1] != self.num_feature):
            self.w = np.insert(self.w, -1, values=0, axis=1)

    def propagate(self, X, Y, num):
        W = self.w[num]
        b = self.b[num]
        y_hat = sigmoid(np.dot(X, W) + b)       
        cost = -(np.sum(Y*np.log(y_hat)+(1-Y)*np.log(1-y_hat))) / self.num_train      
        dZ = y_hat - Y
        dw = np.dot(X.T, dZ) / self.num_train
        db = np.sum(dZ) / self.num_train
        grads = {'dw': dw, 'db': db}
        return grads, cost

    def optimize(self, num, num_iterations, learning_rate, print_cost=False):
        X = self.X_train
        Y = self.Y_train.copy()
        for i in range(len(Y)):
            if Y[i] != num:
                Y[i] = 0
            else:
                Y[i] = 1
        for i in range(num_iterations):
            grads, cost = self.propagate(X, Y, num)
            dw = grads['dw']
            db = grads['db']
            self.w[num] = self.w[num] - learning_rate*dw
            self.b[num] = self.b[num] - learning_rate*db
            if i % 100 == 0:
                if print_cost:
                    print ("Cost after iteration %i for classifier %i: %f" %(i, num, cost))

    def predict(self, X, num):
        m = self.num_feature
        Y_pred = np.zeros((1, m))
        y_hat = sigmoid(np.dot(X, self.w[num])+self.b[num])
        pred = []
        for y in y_hat[:,0:1]:
            pred.append(y[0])
        return np.array(pred)

    def train(self, learning_rate=0.1, num_iterations=1000, print_cost=False, save=True):
        X = self.X_train
        y_hat_train = []
        for i in range(CLASS_NUM):
            self.optimize(i, num_iterations, learning_rate, print_cost)
            y_hat_train.append(self.predict(X, i))
        y_hat_train = np.array(y_hat_train)
        prediction_train = np.argmax(y_hat_train, axis=0)
        cmp_res = np.zeros(len(prediction_train))
        for i in range(len(prediction_train)):
            if prediction_train[i] != self.Y_train[i][0]:
                cmp_res[i] = 1
        accuracy_train = 1 - np.mean(cmp_res)
        #print("Accuracy on train set:", accuracy_train)
        if save:         
            for i in range(CLASS_NUM):
                np.savetxt('w'+str(i)+'.txt', self.w[i])
            np.savetxt('b.txt', self.b)
        return accuracy_train

    def load_parameter(self):
        self.w = np.zeros((CLASS_NUM, self.num_feature, 1))
        self.b = np.zeros(CLASS_NUM)
        for i in range(CLASS_NUM):
            self.w[i] = np.loadtxt('w'+str(i)+'.txt') .reshape((self.num_feature, 1))
        self.b = np.loadtxt("b.txt")


    def test(self, X, Y):       
        y_hat_test = []
        for i in range(CLASS_NUM):
            y_hat_test.append(self.predict(X, i))
        y_hat_test = np.array(y_hat_test)
        prediction_test = np.argmax(y_hat_test, axis=0)
        cmp_res = np.zeros(len(prediction_test))

        for i in range(len(prediction_test)):
            if prediction_test[i] != Y[i]:
                cmp_res[i] = 1

        accuracy_test = 1 - np.mean(cmp_res)
        print("Accuracy on test set:", accuracy_test)
        d = {'w': self.w, 'b': self.b}
        return accuracy_test

if __name__ == '__main__':

    #get_lexicon('train.tsv')

    model = LogisticPredictor('train.tsv')
    model.load_parameter()

    '''
    for i in range(11):
        acc = model.train(learning_rate=0.1, num_iterations=200, print_cost=False, save=True)
        model.read_new_train()
        print("Accuracy on train set %i: %f" %(i, acc))
    '''
    
    #print("Finished loading")
    #print(model.w)
    X_test, Y_test = next(model.generator)
    model.test(X_test, Y_test)
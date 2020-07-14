import numpy as np
import csv
from scipy.sparse import csr_matrix
import string

MAX_ROW_NUM = 1000

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def n_gram_extractor(file_name):
    #for row in dataset:
    #    phrase = Phrase(row.phraseId)
    lexicon = dict()
    Y = np.zeros((MAX_ROW_NUM, 1))
    X = np.zeros((MAX_ROW_NUM, 1))
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
                    X = np.insert(X, -1, values=0, axis=1)
                if token1 not in lexicon:
                    lexicon[token1] = cnt
                    cnt += 1
                    X = np.insert(X, -1, values=0, axis=1)
                if token2 not in lexicon:
                    lexicon[token2] = cnt
                    cnt += 1
                    X = np.insert(X, -1, values=0, axis=1)

                X[row_idx%MAX_ROW_NUM][lexicon[word]] = 1
                X[row_idx%MAX_ROW_NUM][lexicon[token1]] = 1
                X[row_idx%MAX_ROW_NUM][lexicon[token2]] = 1

            Y[row_idx%MAX_ROW_NUM][0] = int(row[3])
            if row_idx % MAX_ROW_NUM == 0:
                yield X, Y

def sigmoid(z):
        return 1 / (1+np.exp(-z))

class LogisticPredictor(object):

    def __init__(self, train_file):
        self.generator = n_gram_extractor('train.tsv')
        self.X_train, self.Y_train = next(self.generator)
        self.num_train = self.X_train.shape[0]
        self.num_feature = self.X_train.shape[1]
        self.w = np.zeros((int(np.max(self.Y_train))+1, self.num_feature, 1))
        self.b = np.zeros(int(np.max(self.Y_train))+1)

    def read_new_train(self):
        self.X_train, self.Y_train = next(self.generator)
        self.num_train = self.X_train.shape[0]
        self.num_feature = self.X_train.shape[1]
        while(self.w.shape[1] != self.num_feature):
            self.w = np.insert(self.w, -1, values=0, axis=1)

    def loss(self, X, Y, num):
        W = self.w[num]
        b = self.b[num]
        y_hat = sigmoid(np.dot(X, W) + b)       
        cost = -(np.sum(Y*np.log(y_hat)+(1-Y)*np.log(1-y_hat))) / self.num_feature       
        dZ = y_hat - Y
        dw = np.dot(X.T, dZ) / self.num_feature
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

        costs = []
        for i in range(num_iterations):
            grads, cost = self.loss(X, Y, num)
            dw = grads['dw']
            db = grads['db']
            #print(dw)
            self.w[num] = self.w[num] - learning_rate*dw
            self.b[num] = self.b[num] - learning_rate*db
            if i % 100 == 0:
                costs.append(cost)
                if print_cost:
                    print ("Cost after iteration %i for classifier %i: %f" %(i, num, cost))

    def predict(self, X, num):
        m = X.shape[1]
        Y_pred = np.zeros((1, m))
        y_hat = sigmoid(np.dot(X, self.w[num])+self.b[num])
        pred = []
        for y in y_hat[:,0:1]:
            pred.append(y[0])
        return np.array(pred)

    def train(self, learning_rate=0.1, num_iterations=1000, print_cost=False):
        y_hat_train = []
        for i in range(int(np.max(self.Y_train))+1):
            self.optimize(i, num_iterations, learning_rate, print_cost)
            y_hat_train.append(self.predict(self.X_train, i))
        y_hat_train = np.array(y_hat_train)
        prediction_train = np.argmax(y_hat_train, axis=0)
        cmp_res = np.zeros(len(prediction_train))
        for i in range(len(prediction_train)):
            if prediction_train[i] != self.Y_train[i][0]:
                cmp_res[i] = 1
        accuracy_train = 1 - np.mean(cmp_res)
        print("Accuracy on train set:", accuracy_train)
        d = {'w': self.w, 'b': self.b}
        return d

    def test(self, X, Y, print_cost=False):
        while(self.w.shape[1] != X.shape[1]):
            self.w = np.insert(self.w, -1, values=0, axis=1)
        y_hat_test = []
        for i in range(int(np.max(Y))+1):
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
        return d

if __name__ == '__main__':

    model = LogisticPredictor('train.tsv')
    model.train(print_cost=False, num_iterations=500)
    model.read_new_train()
    model.train(print_cost=False, num_iterations=500)
    model.read_new_train()
    model.train(print_cost=False, num_iterations=500)
    model.read_new_train()
    model.train(print_cost=False, num_iterations=500)
    model.read_new_train()
    model.train(print_cost=False, num_iterations=500)
    X_test, Y_test = next(model.generator)
    model.test(X_test, Y_test, print_cost=True)
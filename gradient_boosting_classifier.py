from sklearn import utils
from math import sqrt
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import mean_squared_error as mse, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


class gradient_boosting_tree:
    """
    """

    def __init__(self, l: str = 'ls', lr: float = .001, n_e: int = 100, md: int = 10,
                 mss: int = 9, msl: int = 8, c: str = 'friedman_mse', split: float = .1) -> None:
        """
        """
        self.loss = l
        self.learning_rate = lr
        self.n_estimators = n_e
        self.max_depth = md
        self.min_samples_split = mss
        self.min_samples_leaf = msl
        self.criterion = c
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.data_split = split

    def __str__(self) -> str:
        """
        """
        if len(self.y_pred) > 1:
            op = '''
Loss = {}, learning rate = {}, num estimators = {},
max depth = {}, min samples split = {},
min samples leaf = {}, criterion = {},
Size of test data compared to train data = {}.

y prediction = {}
y actual = {}
Mean squared error = {}
root mean squared error = {}
pearson correlation coefficient = {}
Accuracy = {}
Kendall Tau = {}
Win rate = {}'''.format(self.loss, self.learning_rate, self.n_estimators,
                        self.max_depth, self.min_samples_split, self.min_samples_leaf,
                        self.criterion, self.data_split, self.y_pred, self.y_test, mse(
                    self.y_test, self.y_pred),
                        self.RMSE(self.y_pred, self.y_test),
                        pearsonr(self.y_pred, self.y_test.ravel()),
                        self.accuracy(self.y_pred, self.y_test),
                        kendalltau(self.y_pred, self.y_test),
                        str(self.win_rate(self.y_test, self.y_pred) * 100) + '%')

        else:
            op = '''
Loss = {}, learning rate = {}, num estimators = {},
max depth = {}, min samples split = {},
min samples leaf = {}, criterion = {},
Size of test data compared to train data = {}.

y prediction = {}
y actual = {}
Mean squared error = {}
root mean squared error = {}
Win rate = {}'''.format(self.loss, self.learning_rate, self.n_estimators,
                        self.max_depth, self.min_samples_split, self.min_samples_leaf,
                        self.criterion, self.data_split, self.y_pred, self.y_test, mse(
                    self.y_test, self.y_pred),
                        self.RMSE(self.y_pred, self.y_test),
                        str(self.win_rate(self.y_test, self.y_pred) * 100) + '%')
        return op

    def __repr__(self) -> str:
        """
        """
        return self.__str__

    def accuracy(self, ypred, yexact) -> float:
        """
        calculates the accuracy of the predicted vs exact. Likely to be zero
        in the instance of regression
        :param ypred: the predicted data
        :param yexact: the exact data
        :return: the accuracy percentage of the data
        """
        p = np.array(ypred == yexact, dtype=int)
        return np.sum(p) / float(len(yexact))

    def RMSE(self, ypred, yexact) -> float:
        """
        calculates the root mean square error of the data sets
        :param ypred: the predicted dataset
        :param yexact: the exact dataset
        :return: the rmse
        """
        return sqrt(mse(yexact, ypred))

    def win_rate(self, y_test, y_pred) -> float:
        """
        calculates how often we would correctly guess the direction the
        stock price would go
        :param y_test: the exact data
        :param y_pred: the predicted data
        :return: returns the win rate as a percentage
        """
        count = 0
        for i in range(len(y_test)):
            if (y_test[i] > 0.0 and y_pred[i] > 0.0) or \
                    (y_test[i] < 0.0 and y_pred[i] < 0.0) or \
                    (y_test[i] == 0.0 and y_pred[i] == 0.0):
                count += 1
        return count / len(y_test)

    def split_data_classification(self, filename: str = 'NVDA_long.csv') -> None:
        """
        """
        data = pd.read_csv(filename).values
        print('length of data', len(data), ', splits', len(
            data) / 12)  # needs to be a multiple of 12
        i = 0
        x = []
        y_diff = []
        while i in range(len(data)):
            open_prices = []
            high_prices = []
            low_prices = []
            close_prices = []
            volume = []
            averages = []
            maxes = []
            mins = []
            stdevs = []

            for j in range(10):
                open_prices.append(data[i + j][0])
                high_prices.append(data[i + j][1])
                low_prices.append(data[i + j][2])
                close_prices.append(data[i + j][3])
                volume.append(data[i + j][4])

            # 0 for open, 1 for high, 2 for low, 3 for close, 4 for volume
            # average of all only one of lines 140 and 144 can be uncommented
            # y_diff.append(np.average([data[i + 11][0] - data[i + 10][0], data[i + 11][1] - data[i + 10][1], data[i + 11][2] - data[i + 10][2], data[i + 11][3] - data[i + 10][3]]))

            # Just one price
            val = data[i + 11][0] - data[i + 10][0]
            y_diff.append(1 if val > 0 else -1)
            # MAY NEED TO GO UP TO 11TH DAY AND ANALYZE 12-11th PRICE

            averages.append(np.average(open_prices))
            averages.append(np.average(high_prices))
            averages.append(np.average(low_prices))
            averages.append(np.average(close_prices))
            averages.append(np.average(volume))

            maxes.append(np.max(open_prices))
            maxes.append(np.max(high_prices))
            maxes.append(np.max(low_prices))
            maxes.append(np.max(close_prices))
            maxes.append(np.max(volume))

            mins.append(np.min(open_prices))
            mins.append(np.min(high_prices))
            mins.append(np.min(low_prices))
            mins.append(np.min(close_prices))
            mins.append(np.min(volume))

            stdevs.append(np.std(open_prices))
            stdevs.append(np.std(high_prices))
            stdevs.append(np.std(low_prices))
            stdevs.append(np.std(close_prices))
            stdevs.append(np.std(volume))

            lists = [open_prices, high_prices, low_prices, close_prices,
                     volume, averages, maxes, mins, stdevs]
            x_temp = []
            for item in lists:
                x_temp.extend(item)
            x.append(x_temp)
            # print('\nSample {}: {}'.format(i / 12, x_temp))

            i += 12
        x_features = np.array(x)
        y_labels = np.array(y_diff)
        y_labels.reshape(1, -1)

        # split by .1 usually, .001 gives one prediction
        self.x_train, self.x_test, \
        self.y_train, self.y_test = train_test_split(
            x_features, y_labels, test_size=self.data_split)

    def gradient_boost_classify(self) -> None:
        """
        """
        classifier = GradientBoostingClassifier(loss=self.loss,
                                                learning_rate=self.learning_rate,
                                                n_estimators=self.n_estimators,
                                                max_depth=self.max_depth,
                                                min_samples_split=self.min_samples_split,
                                                criterion=self.criterion,
                                                min_samples_leaf=self.min_samples_leaf)

        classifier.fit(self.x_train, self.y_train)
        self.y_pred = classifier.predict(self.x_test)

        print(self)

    def many_classifications(self, num_iter: int = 10) -> None:
        """
        """
        self.n_estimators = 1000
        win_rates = []
        for i in range(1, num_iter + 1):
            print('\nIteration {}:'.format(i))
            self.split_data_classification()
            self.gradient_boost_classify()
            win_perc = self.win_rate(self.y_test, self.y_pred)
            win_rates.append(win_perc * 100)

        print('\nAverage win rate over {} iterations = {}%'.format(
            num_iter, np.average(win_rates)))


def main():
    """
    """
    classifier = gradient_boosting_tree('deviance', .001, 1000, 100,
                                        99, 98, 'friedman_mse', .1)
    classifier.split_data_classification('NVDA_long.csv')
    classifier.gradient_boost_classify()
    # classifier.many_classifications(1001)

if __name__ == '__main__':
    main()
else:
    print('Gradient Boosting Classifier Imported')
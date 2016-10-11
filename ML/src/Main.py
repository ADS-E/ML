from sklearn.svm import SVC

import CsvHelper
import Evaluator
import SetsHelper

data = CsvHelper.divided_by_page()

X_train, X_test, y_train, y_test = SetsHelper.create_sets(data)

clf = SVC(kernel='linear')

Evaluator.train_and_evaluate(clf, X_train, X_test, y_train, y_test)

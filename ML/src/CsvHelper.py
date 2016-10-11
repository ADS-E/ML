import pandas as pd
import numpy as np


def get_raw_data():
    scope = pd.read_csv("scope.csv", delimiter=';')
    no_scope = pd.read_csv("no_scope.csv", delimiter=';')

    scope_length = len(scope)

    data = pd.concat([scope, no_scope], ignore_index=True)
    data['Label'] = 0

    for index, row in data.iterrows():
        label = 1 if index < scope_length else 0
        data.set_value(index, 'Label', label)

    data = data.iloc[np.random.permutation(len(data))]
    print(data)

    return data


def get_data():
    return drop_columns(get_raw_data())


def divided_by_page():
    return divide_by('PageCount')


def divided_by_word():
    return divide_by('WordCount')


def divide_by(value):
    data = get_raw_data()

    for column in data.columns.tolist()[1:-3]:
        data[column] = data[column] / data[value]

    return drop_columns(data)


def drop_columns(data):
    data = data.drop('PageCount', 1)
    data = data.drop('WordCount', 1)

    return data

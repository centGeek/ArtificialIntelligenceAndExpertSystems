import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)


train_data = pd.DataFrame(X_train, columns=iris.feature_names)
train_data['target'] = y_train

test_data = pd.DataFrame(X_test, columns=iris.feature_names)
test_data['target'] = y_test

# train_data.to_csv("train_data.csv", index=False)
# test_data.to_csv("test_data.csv", index=False)

def return_X_train():
    df = pd.read_csv('train_data.csv')
    return df.iloc[:, :-1]
def return_Y_train():
    df = pd.read_csv('train_data.csv')
    return df.iloc[:, -1]

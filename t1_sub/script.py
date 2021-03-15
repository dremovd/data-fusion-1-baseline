import pandas as pd
import pickle
from processing import load_dataset, unique_item_name


test_name = 'data/task1_test_for_user.parquet'
test = load_dataset(test_name, drop_unlabeled=False)
assert 'id' in test.columns

test_unique = unique_item_name(test)

tfidf = pickle.load(open('tfidf', 'rb'))
clf = pickle.load(open('clf_task1', 'rb'))

X_test = tfidf.transform(test_unique.item_name)

pred = clf.predict(X_test)
test_unique['pred'] = pred
test = test.merge(test_unique, on='item_name', how='left')
test[['id', 'pred']].to_csv('answers.csv', index=None)
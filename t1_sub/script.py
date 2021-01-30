import pandas as pd
import pickle


test = pd.read_parquet('data/task1_test_for_user.parquet')
assert 'id' in test.columns

tfidf = pickle.load(open('tfidf', 'rb'))
clf = pickle.load(open('clf_task1', 'rb'))

X_test = tfidf.transform(test.item_name)
pred = clf.predict(X_test)
test['pred'] = pred
test[['id', 'pred']].to_csv('answers.csv', index=None)

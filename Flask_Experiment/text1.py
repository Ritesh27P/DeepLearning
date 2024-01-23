import pickle

model1 = pickle.load(open('Health_Model1.sav', 'rb'))
print(model1)
print(model1.predict([[48, 1, 0, 124, 274, 0, 0, 166, 0, 0.5, 1]]))
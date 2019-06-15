import pandas as pd
import numpy as np
from encode import encode_to_onehot, encode_to_zeor_one, decode_to_emotion
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
train = pd.read_excel("train.xls")
test = pd.read_excel("test.xls")
train = train.values
test = test.values
train_x = train[:, 1]
train_y = train[:, 2]
test_x = test[:, 1]
train_all = np.concatenate((train_x, test_x))
train_y =encode_to_zeor_one(train_y)
train_encode_all = encode_to_onehot(train_all)
train_x = train_encode_all[:6331, :]
test_x = train_encode_all[6331:, :]
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)

model = Sequential()
model.add(
    Dense(
        units=50,
        input_dim=22174,
        activation='sigmoid'
    )
)
model.add(
    Dense(
        units=1,
        activation='sigmoid'
    )
)
# model.add(
#     Dense(
#         units=1,
#         activation='sigmoid'
#     )
# )
sgd = SGD(lr=0.2)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(train_x, train_y, epochs=30, batch_size=32)
result = model.predict(test_x)
# result = decode_to_emotion(result)
result = np.array(result)
result.reshape(-1, 1)
result_csv = pd.DataFrame(result)
result_csv.to_csv('result.csv')










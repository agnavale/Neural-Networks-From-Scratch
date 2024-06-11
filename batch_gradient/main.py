import numpy as np 
from layers import Dense
from models import Sequential

# 2 to 4 decoder , classification problem
X = np.array( [[0, 0], [0, 1], [1, 0], [1, 1]] )
Y = np.array( [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]] )

model = Sequential()
model.add(Dense(2,5,activation="Softmax"))
model.add(Dense(5,4,activation="Softmax"))
model.compile(loss="Cross_entropy", optimizer="SGD", metric= None)
model.fit(X,Y, batch_size=1, epochs=10000)

while True:
    x1 = int(input("x1: "))
    if x1 == -1:
        break
        
    x2 = int(input("x2: "))
    x = np.array([x1,x2])
    print(model.predict(x))


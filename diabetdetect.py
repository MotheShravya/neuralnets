"""The diabetic disease detection usiing cnn majorly for woman, 
factors that may be important to take:
1) how many times pregnant ?
plasma glucose concentration on 2 hours in oral glucose 
distolic blood pressure (mm hg)
triceps skin fold thickness
2 hour serun insulin
body mass indexx
diabetes pedgree function
age
class variable"""

from numpy import loadtxt
from keras.models import Sequential , model_from_json
from keras.layers import Dense

dataset =  loadtxt('pima-indians-diabetes.csv',delimiter= ',')
x = dataset[:,0:8]
y = dataset[:,8]
 
model = Sequential()
model.add(Dense(12, input_dim = 8,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics = ['accuracy'] )
model.fit(x,y,epochs = 10, batch_size= 10)
_, accuracy = model.evaluate(x,y)
print('accuracy :  %.2f' % (accuracy*100))

model_jsn = model.to_json()
with open("model.json",'w')as json_file:
    json_file.write(model_jsn)
model.save_weights("model.h5")
print('saved models')
pred = model.predict(x)
for i in range(5,10):
    print('%s=> %d(expected %d)' %(x[i].tolist(),pred[i],y[i]))
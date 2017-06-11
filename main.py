import numpy as np

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

np.random.seed(42)

(x_train, y_train),(x_test, y_test) = cifar10.load_data()



#нормализация данных по интенсивности пикселей
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /=255
x_test /=255

print(y_train[1])
#приводим метки класса к категориальному виду:
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

#создаем модель:
model = Sequential()

#добавляем слои:

#ПЕРВЫЙ КАСКАД СЛОЁВ:
#первый сверточный слой
model.add(Convolution2D(32,3,3, border_mode="same", input_shape=(32,32,3), activation="relu"))
#второй сверточный слой:
model.add(Convolution2D(32,3,3, activation="relu")) #32,3,3 - карта свертки
#слой подвыборки:
model.add(MaxPooling2D(pool_size=(3,3), dim_ordering="th")) #из 2Х2 выбирается максимальное значение -MaxPooling2D
#слой регуляции:
model.add(Dropout(0.25)) # - dropout - отключение некоторых нейронов во избежание переобучения, 25% вероятность отключения, для 1 нейрона

#ВТОРОЙ КАСКАД СЛОЁВ:
#первый сверточный слой
model.add(Convolution2D(64,3,3, border_mode="same", input_shape=(32,32,3), activation="relu")) #увеличиваем количество карт свертки
#второй сверточный слой:
model.add(Convolution2D(64,3,3, activation="relu")) 
#слой подвыборки:
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
#слой регуляции:
model.add(Dropout(0.25))

#преобразователь из двумерного представления в плоское:
model.add(Flatten())
#полносвязный слой:
model.add(Dense(512, activation="relu"))
#слой регуляции:
model.add(Dropout(0.5))

#ВЫХОДНОЙ СЛОЙ:
model.add(Dense(10, activation="softmax"))


#компиляция модели
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

#обучение:
model.fit(x_train,y_train,batch_size=32, nb_epoch=30, validation_split=0.1, shuffle=True)#shuffle -перемешивание данных в начале каждой эпохи

# проверка точности:
scores = model.evaluate(x_test,y_test, verbose=0)
print(r"Точность: ", (scores[1]*100))


#Работа Модели:
prediction = model.predict(x_train) 
prediction = prediction.round(0) # определяем лидирующий нейрон, подходит так как использован "softmax" 
# print(prediction[:1])
# print(y_train[:1]) 

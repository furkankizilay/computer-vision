import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

imgs = glob.glob("C:/Users/furka/Desktop/computer-vision/cnn-with-opencv/trex-project/img_/*.png") # get the image

width = 125
height = 50 

x = [] 
y = []

for img in imgs:
    filename = os.path.basename(img)
    label = filename.split("_")[0]
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im / 255
    x.append(im)
    y.append(label)

X = np.array(x)
X = X.reshape(X.shape[0], width, height, 1)

# sns.countplot(y)

def oneHotLabels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values) # 0 1 2
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded) # 001, 010, 100
    return onehot_encoded

Y = oneHotLabels(y)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=2)

# cnn model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation = "relu", input_shape = (width, height, 1)))
model.add(Conv2D(64, kernel_size=(3,3), activation = "relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(3, activation="softmax"))

"""if os.path.exists("./trex_weight.h5"):
    model.load_weights("trex_weight.h5")"""

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["Accuracy"])
model.fit(train_X, train_Y, epochs=35, batch_size=64) # epochs: how many times to train in total, batch size: how many groups of pictures will enter an iteration 

score_train = model.evaluate(train_X, train_Y)
print("Train Score: %", score_train[1]*100)
score_test = model.evaluate(test_X, test_Y)
print("Test Score: %", score_test[1]*100)

open("C:/Users/furka/Desktop/computer-vision/cnn-with-opencv/trex-project/model.json","w").write(model.to_json())
model.save_weights("C:/Users/furka/Desktop/computer-vision/cnn-with-opencv/trex-project/trex_weight.h5")







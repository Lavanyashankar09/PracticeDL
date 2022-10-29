from tensorflow import keras
import numpy as np

fashion_mnist=keras.datasets.fashion_mnist

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

X_train=X_train/255.0
X_test=X_test/255.0

X_train=X_train.reshape(len(X_train),28,28,1)
X_test=X_test.reshape(len(X_test),28,28,1)

def build_model(hp):  
  model = keras.Sequential([
      
    keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(28,28,1)
    ),
    
    keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ),
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    
    keras.layers.Dense(10, activation='softmax')
  ])
  
  model.compile(optimizer=keras.optimizers.Adam(
              hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  return model
  
from keras_tuner import RandomSearch

tuner=RandomSearch(build_model,
                          objective='val_accuracy',
                          max_trials=5,directory='output',project_name="Mnist Fashion")

tuner.search_space_summary()

tuner.search(X_train,y_train,epochs=2,validation_split=0.1)

tuner.results_summary()


model=tuner.get_best_models(num_models=1)[0]

model.summary()

model.fit(X_train, y_train, epochs=10, validation_split=0.1, initial_epoch=2)

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis = 1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred_classes) 

a = accuracy_score(y_test, y_pred_classes) 



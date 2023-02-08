#!/usr/bin/env python
# coding: utf-8

# # Commonly Used tf.Keras Functionality

# This notebooks contains commonly used `tf.keras` functionality to develop a neural network model using `keras` as the API and `tensorflow` as the backend.

# ## Importing Modules

# In[27]:


import tensorflow as tf
from tensorflow import keras
import numpy as np


# In[28]:


keras.__version__


# ## Example: Building a Classifier and Regressor Using the Sequential API

# #### **Importing the Dataset using `tf.keras.datasets`**

# In[29]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# In[30]:


X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# #### **Creating the model using the Sequential API**

# In[31]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()


# In[32]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


# Simple plot of training history:

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

history_df = pd.DataFrame(history.history)
history_df.plot(figsize=(10, 6))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# Evaluate the model using testing set:

# In[34]:


model.evaluate(X_test, y_test)


# #### **Creating the model using the Functional API**

# Keras' `Functional API` provides a more flexible way of creating a neural network model. We will try to develop a *Wide & Deep* neural network using Keras' Functional API.

# ![](assets\wide-deep-model.png)
# *source:  [Papers With Code](https://paperswithcode.com/method/wide-deep)
# 

# In[35]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# fetching California Dataset
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
    )
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full
    )
# scaling the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
# functional API
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input_)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])
# compiling model
model.compile(loss='mse', optimizer=keras.optimizers.SGD(learning_rate=1e-4))
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


# In[36]:


model.evaluate(X_test, y_test)


# #### **Multi Input/Output NN using Subclassing API**

# Even more flexible model using `Subclassing API`:

# In[37]:


class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

# dataset
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

# model
model = WideAndDeepModel()
input_A = keras.layers.Input(shape=[5], name='wide_input')
input_B = keras.layers.Input(shape=[6], name='deep_input')
model.call([input_A, input_B])
# compiling model
model.compile(
    loss=['mse', 'mse'], loss_weights=[0.9, 0.1],
    optimizer='sgd'
    )
history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=20,
    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))


# In[38]:


X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])
print(
    f'total loss: {total_loss:.2f}\n'
    f'main_loss: {main_loss:.2f}\n'
    f'aux_loss: {aux_loss:.2f}'
)


# #### **Early Stopping using keras.callbacks.EarlyStopping**
# 

# Keras provides various callbacks that can be implemented to improve our neural net learning process. In this example, we will implement `EarlyStopping`:

# In[39]:


early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=200,
    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]),
    callbacks=[early_stopping_cb]
    )


# #### **Utilizing TensorBoard for Visualization Aid**

# Tensorflow provides `TensorBoard` for a visualization of training history:

# In[40]:


import os
root_logdir = os.path.join(os.curdir, 'run_logs')

def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=200,
    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]),
    callbacks=[early_stopping_cb, tensorboard_cb], verbose=0
    )


# Let's create another model to create comparative visualization of different optimizers.

# In[41]:


model.compile(
    loss=['mse', 'mse'], loss_weights=[0.9, 0.1],
    optimizer='adam'
    )
tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())
history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=200,
    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]),
    callbacks=[early_stopping_cb, tensorboard_cb], verbose=0
    )


# To run tensorboard on local port:

# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir=./run_logs')


# #### **Hyperparameter Tuning using Keras' Sk-Learn Wrapper**

# We will use scikit-learn interfaces for randomized search cross validation for hyperparameter tuning.

# In[17]:


# creating model function, specify parameters to be cv-ed
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, activation='relu'):
    input_ = keras.layers.Input(shape=[8])
    input_A = keras.layers.Lambda(lambda x: x[:, :5])(input_)
    input_B = keras.layers.Lambda(lambda x: x[:, 2:])(input_)
    dense = keras.layers.Dense(n_neurons, activation=activation)(input_B)
    for layer in range(n_hidden - 1):
        dense = keras.layers.Dense(n_neurons, activation=activation)(dense)
    wide_deep = keras.layers.Concatenate(axis=1)([input_A, dense]) # axis=1 to concat horizontally
    main_output = keras.layers.Dense(1)(wide_deep)
    aux_output = keras.layers.Dense(1)(wide_deep)
    model = keras.Model(inputs=[input_], outputs=[main_output, aux_output])
    # compiling model
    model.compile(
        loss=['mse', 'mse'], loss_weights=[0.9, 0.1], 
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model
# sklearn wrapper
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
# define fit parameter
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
keras_reg.fit(
    X_train, y_train, epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping_cb], verbose=0
    )
mse_test = keras_reg.score(X_test, y_test)


# In[18]:


from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    'n_hidden': [1, 2, 3, 4],
    'n_neurons': np.arange(1, 100),
    'learning_rate': reciprocal(3e-4, 3e-2)
    }

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(
    X_train, y_train, epochs=500,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping_cb], verbose=0
    )
print(f'best params: {rnd_search_cv.best_params_}')
print(f'best score: {rnd_search_cv.best_score_}')

model = rnd_search_cv.best_estimator_.model


# ## Tensorflow's Tensor Operations

# ### *Constant*

# Constant is an immutable data type, hence can be used for parameters that need to change over time (for example: neuron weights).
# 
# Creating tensor *constant* from python object:

# In[19]:


t = tf.constant([[1, 2, 3], [4, 5, 6]])
t


# Creating tensor using numpy array:

# In[23]:


a = np.array([[1, 2, 3], [4, 5, 6]])
t = tf.constant(a)
t


# We should carefully consider about the datatypes of the tensors:

# In[26]:


b = tf.constant(1) 
print(f'tensor "b", datatypes: {b.dtype}')
c = tf.constant(1.0)
print(f'tensor "c", datatypes: {c.dtype}')
b + c


# We can use `tf.cast` to cast the datatype:

# In[29]:


b + tf.cast(c, b.dtype)


# ### *Variables*

# Another tensorflow datatypes which is mutable.

# In[33]:


v = tf.Variable([[1, 2, 3], [4, 5, 6]])
v


# We can modify the variable in place:

# In[34]:


v.assign(2 * v)
print(v)
v[0, 1].assign(42)
print(v)
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100, 200])
print(v)


# ## Keras' Preprocessing Layer

# Keras provides preprocessing layers that worth to be considered. Here is an example of `tf.keras.layers.Normalization`:

# In[48]:


a = tf.constant(X_train_A[:, 1].reshape(-1, 1))
# normalization layer
norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(a)
a_norm = norm_layer(a)


# ## Reference

# This notebook provides some examples from `Hands on Machine Learning with Scikit-learn, Keras, and Tensorflow` by Aurelion Geron.

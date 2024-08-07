#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv1D,Add
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool1D,Concatenate
from tensorflow.keras.layers import GlobalAvgPool1D
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical


# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("E:/new_transformer/trial.csv")

x = df.iloc[:, 0:21]
y = df.iloc[:, 21]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .90)


# In[3]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform (x_test)


# In[4]:


x_train_reshaped = tf.reshape(x_train, (-1, 21, 1))
x_test_reshaped = tf.reshape(x_test, (-1, 21, 1))
print(x_train_reshaped.shape)
print(x_test_reshaped.shape)


# In[134]:


#no
x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[121]:


#no
n_features = 1
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
x_train.shape


# In[5]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense

input_shape = (21,)

# Create an input layer with the defined shape
inputs = tf.keras.layers.Input(shape=input_shape)

from keras import regularizers
regularization_value = 0.01
regularizer = tf.keras.regularizers.l2(regularization_value)

# Add 1D CNN layers
x = tf.keras.layers.Reshape((21, 1))(inputs)
x = Conv1D(filters=64, kernel_size=3, padding = 'same',activation='relu',kernel_regularizer=regularizer, bias_regularizer=regularizer)(x) #,input_shape = (62000,21,500)
#x = Reshape((62000,21))(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=128, kernel_size=3, activation='relu',kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=256, kernel_size=3, activation='relu',kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
x = MaxPooling1D(pool_size=2)(x)


# Add dense and dropout layers
x = Dense(64, activation='relu',kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
#x = Dropout(0.5)(x)
x = Dense(21, activation='relu',kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)

# Define the model
model = Model(inputs=inputs, outputs=x)

# Show model summary
model.summary()


# In[60]:



# In[8]:


maxlen = 7      # Only consider 3 input time points
d_model = 3
num_heads = 8
dff = 64


# In[9]:


input_tran = x
embedding_layer = TokenAndPositionEmbedding(maxlen, d_model)
x1 = embedding_layer(input_tran)


# In[10]:


#new
from keras import regularizers
regularization_value = 0.01
regularizer = tf.keras.regularizers.l2(regularization_value)

# Encoder Architecture
transformer_block_1 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)
transformer_block_2 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)
transformer_block_3 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)
transformer_block_4 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)




# In[64]:


# Encoder Architecture
transformer_block_1 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff)
transformer_block_2 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff)
transformer_block_3 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff)
#transformer_block_4 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff)



# In[11]:


# Output
x = layers.GlobalMaxPooling1D()(x)
#x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu",kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
#x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)


# In[12]:


model = Model(inputs=inputs, outputs=outputs)


# In[13]:


model.summary()


# In[14]:


model = Model (inputs, outputs)
model.summary()


# In[15]:


from tensorflow.keras import optimizers
#optimizer=keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-05),
              loss="binary_crossentropy",
              metrics=['accuracy'])
model.summary()


# In[ ]:





# In[89]:


#no
import numpy as np

# Assuming you already have X_train as a 2D array of shape (num_samples, num_features)

# Expand dimensions of X_train to make it 3D
X_train_3d = np.expand_dims(x_train, axis=1)

# Reshape X_train_3d to have the desired shape (None, 62000, 21)
X_train_3d = np.reshape(X_train_3d, (-1, x_train.shape[0], x_train.shape[1]))

# Now X_train_3d has the shape (None, 62000, 21)


# In[117]:


#no
import numpy as np

# Assuming you already have X_train as a 2D array of shape (num_samples, num_features)

# Expand dimensions of X_train to make it 3D
X_train_3d1 = np.expand_dims(x_train, axis=2)

# Reshape X_train_3d to have the desired shape (None, 62000, 21)
X_train_3d = np.transpose(X_train_3d1, (0, 1, 2))

# Now X_train_3d has the shape (None, 62000, 21)
X_train_3d1.shape,X_train_3d.shape


# In[ ]:


#no
x_train_reshaped = x_train.reshape(x_train.shape[1], x_train.shape[2], x_train.shape[1])


# In[16]:


#no
history = model.fit(x_train_reshaped, y_train, epochs=100, batch_size=16, validation_data=(x_test_reshaped, y_test))


# In[16]:


import keras
from matplotlib import pyplot as plt
history = model.fit(x_train_reshaped, y_train, epochs=80, batch_size=16, validation_data=(x_test_reshaped, y_test))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.ylim(0, 100)  # Set the y-axis limits to 0 and 100
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[17]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.ylim(0, 100)  # Set the y-axis limits to 0 and 100
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[18]:


model.evaluate(x_test_reshaped, y_test)


# In[19]:


model.predict(x_test)


# In[20]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

predictions = model.predict(x_test)

# Convert predictions to binary class labels (if needed)
binary_predictions = (predictions > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, binary_predictions)
print("Accuracy:", accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_test, binary_predictions)
print("Confusion Matrix:")
print(cm)


# In[79]:


pip install seaborn


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(3, 2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[ ]:


total1=sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)


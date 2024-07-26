#!/usr/bin/env python
# coding: utf-8

# In[21]:


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


# In[22]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("E:\\New_arithmetic_task - Copy\\TRANSFORMER\\SUB-25.csv")

x = df.iloc[:, 0:21]
y = df.iloc[:, 21]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .90)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform (x_test)


# In[24]:


x_train_reshaped = tf.reshape(x_train, (-1, 21, 1))
x_test_reshaped = tf.reshape(x_test, (-1, 21, 1))
print(x_train_reshaped.shape)
print(x_test_reshaped.shape)


# In[25]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense,GRU

# Define input shape

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


# In[26]:




# In[28]:


maxlen = 7      # Only consider 3 input time points
d_model = 3
num_heads = 8
dff = 64


# In[29]:


input_tran = x
embedding_layer = TokenAndPositionEmbedding(maxlen, d_model)
x1 = embedding_layer(input_tran)


# In[30]:


#new
from keras import regularizers
regularization_value = 0.01
regularizer = tf.keras.regularizers.l2(regularization_value)

# Encoder Architecture
transformer_block_1 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)
transformer_block_2 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)
transformer_block_3 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)
transformer_block_4 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)




# In[31]:


# Output
x = layers.GlobalMaxPooling1D()(x)
#x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu",kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
#x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)


# In[32]:


model = Model(inputs=inputs, outputs=outputs)
model.summary()


# In[33]:


from tensorflow.keras import optimizers
#optimizer=keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-05),
              loss="binary_crossentropy",
              metrics=['accuracy'])
model.summary()


# In[34]:


import keras
from matplotlib import pyplot as plt
history = model.fit(x_train_reshaped, y_train, epochs=100, batch_size=16, validation_data=(x_test_reshaped, y_test))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.ylim(0, 100)  # Set the y-axis limits to 0 and 100
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[35]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.ylim(0, 100)  # Set the y-axis limits to 0 and 100
plt.show()


# In[36]:


model.evaluate(x_test_reshaped, y_test)


# In[37]:


model.predict(x_test_reshaped)


# In[38]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

predictions = model.predict(x_test_reshaped)

# Convert predictions to binary class labels (if needed)
binary_predictions = (predictions > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, binary_predictions)
print("Accuracy:", accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_test, binary_predictions)
print("Confusion Matrix:")
print(cm)
report = classification_report(y_test, binary_predictions)
print(report)


# In[39]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(3, 2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[40]:


total1=sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)


# In[ ]:





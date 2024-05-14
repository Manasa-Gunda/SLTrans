#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("E:/new_transformer/trial.csv")

x = df.iloc[:, 0:21]
y = df.iloc[:, 21]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .90)


# In[4]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform (x_test)


# In[5]:


x_train_reshaped = tf.reshape(x_train, (-1, 21, 1))
x_test_reshaped = tf.reshape(x_test, (-1, 21, 1))
print(x_train_reshaped.shape)
print(x_test_reshaped.shape)


# In[6]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense,SimpleRNN

# Define input shape

input_shape = (21,)

# Create an input layer with the defined shape
inputs = tf.keras.layers.Input(shape=input_shape)

from keras import regularizers
regularization_value = 0.001
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

# Add LSTM layer
x = SimpleRNN(128, return_sequences=True,kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
x = Flatten()(x)

# Add dense and dropout layers
x = Dense(64, activation='relu',kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
#x = Dropout(0.5)(x)
x = Dense(21, activation='relu',kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)

# Define the model
model = Model(inputs=inputs, outputs=x)

# Show model summary
model.summary()


# In[7]:


#new
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.5, kernel_regularizer=None, bias_regularizer=None):
        super(TransformerBlock, self).__init__()

        self.multi_head_attention1 = layers.MultiHeadAttention(d_model, num_heads,
                                                       kernel_regularizer=regularizer,bias_regularizer=regularizer)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

        self.simplernn = layers.SimpleRNN(d_model, return_sequences=True,
                         kernel_regularizer=regularizer,bias_regularizer=regularizer)

        self.dropout1 = layers.Dropout(rate)
        self.dense1 = layers.Dense(dff, activation='relu',
                            kernel_regularizer=regularizer,bias_regularizer=regularizer)
        self.dense2 =layers.Dense(d_model,
                            kernel_regularizer=regularizer,bias_regularizer=regularizer)

        self.dropout2 = layers.Dropout(rate)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                                             

    def call(self, inputs, training=False):
        # Apply multi-head attention and add residual connection
        inputs = tf.keras.layers.Reshape((7, 3))(inputs)
        attn_output = self.multi_head_attention1(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        simplernn_output = self.simplernn(out1)
        #lstm_output = self.dropout1(lstm_output, training=training)
        out2 = self.layernorm1(out1 + simplernn_output)  # modify addition
        
        simplernn_output1 = self.simplernn(out2)
        #lstm_output1 = self.dropout1(lstm_output1, training=training)
        out21 = self.layernorm1(out2 + simplernn_output1)  # modify addition
        
   
        # Apply feedforward network and add residual connection
        ff_output = self.dense1(out21)
        ff_output = self.dense2(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        #ff_output = self.lstm(ff_output)
        out3 = self.layernorm2(out21 + ff_output)
        
        return out3


# In[8]:


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, d_model):
        super(TokenAndPositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = tf.reshape(x, [-1, maxlen, d_model])
        out = x + positions
        return out


# In[9]:


maxlen = 7      # Only consider 3 input time points
d_model = 3
num_heads = 8
dff = 64


# In[10]:


input_tran = x
embedding_layer = TokenAndPositionEmbedding(maxlen, d_model)
x1 = embedding_layer(input_tran)


# In[12]:


#new
from keras import regularizers
regularization_value = 0.001
regularizer = tf.keras.regularizers.l2(regularization_value)

# Encoder Architecture
transformer_block_1 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)
transformer_block_2 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)
transformer_block_3 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)
transformer_block_4 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff,kernel_regularizer=regularizer, bias_regularizer=regularizer)

x = transformer_block_1(x)
x = transformer_block_2(x)
x = transformer_block_3(x)
x = transformer_block_4(x)


# In[13]:


# Output
x = layers.GlobalMaxPooling1D()(x)
#x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu",kernel_regularizer=regularizer)(x)#, bias_regularizer=regularizer)(x)
#x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)


# In[14]:


model = Model(inputs=inputs, outputs=outputs)
model.summary()


# In[15]:


from tensorflow.keras import optimizers
#optimizer=keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-05),
              loss="binary_crossentropy",
              metrics=['accuracy'])
model.summary()


# In[ ]:


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


# In[19]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.ylim(0, 100)  # Set the y-axis limits to 0 and 100
plt.show()


# In[20]:


model.evaluate(x_test_reshaped, y_test)


# In[21]:


model.predict(x_test)


# In[25]:


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


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(2, 2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[ ]:


#Confusion matrix, Accuracy, sensitivity, and specificity
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(test_df[['test']],predicted_class1)
print('Confusion Matrix :', cm)


# In[27]:


total1=sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])

y1 = [99.03, 99.75, 98.75, 99.32, 99.77, 99.69, 99.70, 98.9, 
                        99.61, 98.35, 99.46, 96.61, 98.72, 99.64, 99.50, 
                        99.43, 97.29, 98.58, 98.96, 98.35, 99.67, 91.67, 
                        98.32, 98.59, 98.35, 98.00, 95.85, 98.79, 99.70, 
                        97.90, 99.11, 99.50, 98.37, 99.37, 99.12, 99.38]

y2 = [98.75, 97.00, 99.22, 99.52, 99.82, 99.82, 99.64, 
                       98.74, 99.83, 98.30, 98.91, 96.90, 98.79, 99.66, 
                       98.54, 99.33, 92.27, 88.04, 98.96, 98.77, 97.67, 
                       99.20, 99.00, 98.53, 99.70, 98.5, 96.54, 99.62, 
                       99.75, 96.75, 99.18, 99.77, 99.30, 99.04, 95.50, 99.51]

y3 = [98.66, 96.96, 94.93, 98.93, 99.82, 99.70, 97.59, 
                             91.83, 99.53, 94.58, 99.24, 97.38, 98.01, 97.98, 
                             99.09, 97.96, 96.74, 91.87, 98.46, 97.69, 95.19, 
                             98.80, 99.08, 97.54, 95.62, 95.66, 97.03, 98.96, 
                             99.26, 87.95, 98.67, 92.27, 98.95, 95.83, 85.38, 99.30]


plt.figure(figsize=(10,5))
plt.plot(x ,y1 , marker='o', linestyle='-', color='blue',label = 'CNN+LSTM+SLTrans')
plt.plot(x ,y2, marker='s', linestyle='-', color='red',label = 'CNN+GRU+SLTrans')
plt.plot(x ,y3, marker='v', linestyle='-', color='green',label = 'CNN+SimpleRNN+SLTrans')
#plt.figure(figsize=(10, 6), dpi=150)
plt.xlabel('Students',size=10)
plt.ylabel('Accuracy(%)',size=10)
plt.title('Accuracy Output of Proposed Model:Three Techniques',size=10)

plt.legend()
#plt.xticks(x, [str(i) for i in y1,y2,y3], rotation=90)

#set parameters for tick labels
#plt.tick_params(axis='x', which='major', labelsize=3)
#plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.xticks(range(0,36,1),rotation=0)

#plt.tight_layout()
plt.show()


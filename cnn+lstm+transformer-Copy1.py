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

# Add LSTM layer
x = LSTM(128, return_sequences=True,kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
x = Flatten()(x)

# Add dense and dropout layers
x = Dense(64, activation='relu',kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
#x = Dropout(0.5)(x)
x = Dense(21, activation='relu',kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)

# Define the model
model = Model(inputs=inputs, outputs=x)

# Show model summary
model.summary()


# In[60]:


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super(TransformerBlock, self).__init__()
        
        self.multi_head_attention1 = layers.MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        
        self.lstm = layers.LSTM(d_model, return_sequences=True)  # modify LSTM layer
        
        self.dropout1 = layers.Dropout(rate)
        self.dense1 = layers.Dense(dff, activation='relu')
        self.dense2 = layers.Dense(d_model)
        
        self.dropout2 = layers.Dropout(rate)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, training=False):
        # Apply multi-head attention and add residual connection
        attn_output = self.multi_head_attention1(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Apply LSTM and add residual connection
        lstm_output = self.lstm(out1)
        lstm_output = self.dropout1(lstm_output, training=training)
        out2 = self.layernorm1(out1 + lstm_output)  # modify addition
        
        #lstm_output1 = self.lstm(out2)
        #lstm_output1 = self.dropout1(lstm_output1, training=training)
        #out21 = self.layernorm1(out2 + lstm_output1)  # modify addition
        
   
        # Apply feedforward network and add residual connection
        ff_output = self.dense1(out2)
        ff_output = self.dense2(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        ff_output = self.lstm(ff_output)
        out3 = self.layernorm2(out2 + ff_output)
        
        return out3


# In[6]:


#new
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.5, kernel_regularizer=None, bias_regularizer=None):
        super(TransformerBlock, self).__init__()

        self.multi_head_attention1 = layers.MultiHeadAttention(d_model, num_heads,
                                                       kernel_regularizer=regularizer,bias_regularizer=regularizer)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

        self.lstm = layers.LSTM(d_model, return_sequences=True,
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

        lstm_output = self.lstm(out1)
        #lstm_output = self.dropout1(lstm_output, training=training)
        out2 = self.layernorm1(out1 + lstm_output)  # modify addition
        
        lstm_output1 = self.lstm(out2)
        #lstm_output1 = self.dropout1(lstm_output1, training=training)
        out21 = self.layernorm1(out2 + lstm_output1)  # modify addition
        
   
        # Apply feedforward network and add residual connection
        ff_output = self.dense1(out21)
        ff_output = self.dense2(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        #ff_output = self.lstm(ff_output)
        out3 = self.layernorm2(out21 + ff_output)
        
        return out3


# In[7]:


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

x = transformer_block_1(x)
x = transformer_block_2(x)
x = transformer_block_3(x)
x = transformer_block_4(x)


# In[64]:


# Encoder Architecture
transformer_block_1 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff)
transformer_block_2 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff)
transformer_block_3 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff)
#transformer_block_4 = TransformerBlock(d_model=d_model, num_heads=num_heads, dff=dff)

x1 = transformer_block_1(x1)
x1 = transformer_block_2(x1)
x1 = transformer_block_3(x1)
#x1 = transformer_block_4(x1)


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


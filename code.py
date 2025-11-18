# Commented out IPython magic to ensure Python compatibility.
# Load, explore and plot data
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# %matplotlib inline

# Train test split
from sklearn.model_selection import train_test_split

# Text pre-processing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.callbacks import EarlyStopping 

# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/train.csv', encoding='ISO-8859-1') 
# rename the columns
df = df[['text','target']]
df.rename(columns={'text':'text', 'target':'target'}, inplace=True) 
df.head()

df.describe()

print(df.columns) 

df.groupby('target').describe().T 

ham_msg_text = ' '.join(df['text'].dropna().astype(str).tolist()) 
ham_msg_cloud = WordCloud(width =520, height =260, stopwords = STOPWORDS, max_font_size = 50, background_color = "black", colormap = 'Pastel1').generate(ham_msg_text)
plt.figure(figsize=(16,10))
plt.imshow(ham_msg_cloud, interpolation = 'bilinear')
plt.axis('off') 
plt.show()

#Get length column for each text and convert the text label to numeric value:
df['text_length'] = df['text'].apply(len)
df['msg_type'] = df['target'].map({0:0, 1:1})
msg_label =df['msg_type'].values
df.head()

"""**Train - Test split** (80 - 20 ratio)"""

x_train, x_test, y_train, y_test = train_test_split(df['text'], msg_label, test_size=0.2, random_state=434)

"""**Tokenization-sentence ke word e break down kore, lowercase kore, removes punctuation**"""


max_len = df['text'].apply(len).max()
# Output the max length
print(max_len)

# Defining pre-processing parameters
max_len = max_len
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' # out of vocabulary token
vocab_size = 500

tokenizer = Tokenizer(num_words = vocab_size,
                      char_level = False,
                      oov_token = oov_tok)
tokenizer.fit_on_texts(x_train)

# Get the word_index
word_index = tokenizer.word_index
total_words = len(word_index)
total_words

"""**Sequence and** **padding** (ekek sentence ekek length hoy, so maximum length of sentence ke max_length dhore choto size sentence gulake padding kore)"""

training_sequences = tokenizer.texts_to_sequences(x_train) 
training_padded = pad_sequences(training_sequences,
                                maxlen = max_len,
                                padding = padding_type,
                                truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences,
                               maxlen = max_len,
                               padding = padding_type,
                               truncating = trunc_type)

print('Shape of training tensor: ', training_padded.shape)
print('Shape of testing tensor: ', testing_padded.shape)

"""**Using LSTM first**"""

# Define parameter , egula hyperparameters
vocab_size = 19416
embedding_dim = 50
drop_value = 0.2
n_dense = 24

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
embedding_matrix = create_embedding_matrix('/content/glove.6B.50d.txt',tokenizer.word_index, embedding_dim)

# Define parameter
n_lstm = 128
drop_lstm = 0.2
# Define LSTM Model
model1 = Sequential()
model1.add(Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=max_len,
                           trainable=False))
model1.add(SpatialDropout1D(drop_lstm))
model1.add(LSTM(n_lstm, return_sequences=False))
model1.add(Dropout(drop_lstm))
model1.add(Dense(1, activation='sigmoid'))
model1.build(input_shape =( None, max_len))

"""**Compile the model**"""

model1.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

model1.summary()

"""**Train the model**"""

num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = model1.fit(training_padded,
                     y_train,
                     epochs=num_epochs,
                     validation_data=(testing_padded, y_test),
                     callbacks =[early_stop],
                     verbose=2)

"""**Plotting graph of accuracy of train and validation**"""

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)
# Plotting er code
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.ylim([0, 1])
plt.legend()
plt.grid()
plt.show()

"""**Plotting graph of loss of train and validation**"""

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)
# Plotting er code
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.ylim([0, max(max(train_loss), max(val_loss)) * 1.1]) 
plt.legend()
plt.grid()
plt.show()

"""**Printing the accuracy of train and valid data:**"""

final_train_accuracy1 = history.history['accuracy'][-1]
final_val_accuracy1 = history.history['val_accuracy'][-1]
print(f"Training Accuracy: {final_train_accuracy1:.4f}")
print(f"Validation Accuracy: {final_val_accuracy1:.4f}")

"""**Using Bi-LSTM**"""

# Define parameter , egula hyperparameters
vocab_size = 19416
embedding_dim = 50
drop_value = 0.2
n_dense = 24

embedding_matrix = create_embedding_matrix('/content/glove.6B.50d.txt',tokenizer.word_index, embedding_dim)

# Define parameter
n_lstm = 128
drop_lstm = 0.2
model2 = Sequential()
model2.add(Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=max_len,
                           trainable=False))
model2.add(Bidirectional(LSTM(n_lstm,
                              return_sequences = False)))
model2.add(Dropout(drop_lstm))
model2.add(Dense(1, activation='sigmoid'))
model2.build(input_shape =( None, max_len))

"""**Compile the model**"""

model2.compile(loss = 'binary_crossentropy',
               optimizer = 'adam',
               metrics=['accuracy'])

model2.summary()

"""**Train model**"""

num_epochs = 30
early_stop = EarlyStopping(monitor = 'val_loss',
                           patience = 2)
history = model2.fit(training_padded,
                     y_train,
                     epochs = num_epochs,
                     validation_data = (testing_padded, y_test),
                     callbacks = [early_stop],
                     verbose = 2)

"""**Plotting graph of accuracy of train and validation**"""

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)
# Plotting er c
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.ylim([0, 1])
plt.legend()
plt.grid()
plt.show()

"""**Plotting graph of loss of train and validation**"""

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)
# Plotting er code
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.ylim([0, max(max(train_loss), max(val_loss)) * 1.1])  
plt.legend()
plt.grid()
plt.show()

"""**Printing the accuracy of train and valid data:**"""

final_train_accuracy2 = history.history['accuracy'][-1]
final_val_accuracy2 = history.history['val_accuracy'][-1]
print(f"Training Accuracy: {final_train_accuracy2:.4f}")
print(f"Validation Accuracy: {final_val_accuracy2:.4f}")

"""**Using GRU**"""

# Define parameter , egula hyperparameters
vocab_size = 19416
embedding_dim = 50
drop_value = 0.2
n_dense = 24

embedding_matrix = create_embedding_matrix('/content/glove.6B.50d.txt',tokenizer.word_index, embedding_dim) 

model3 = Sequential()
model3.add(Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=max_len,
                           trainable=False))
model3.add(SpatialDropout1D(0.2))
model3.add(GRU(128, return_sequences = False))
model3.add(Dropout(0.2))
model3.add(Dense(1, activation = 'sigmoid'))
model3.build(input_shape =( None, max_len))

"""**Compile the model**"""

model3.compile(loss = 'binary_crossentropy',
               optimizer = 'adam',
               metrics=['accuracy'])

model3.summary()

"""**Train model**"""

num_epochs = 30
early_stop = EarlyStopping(monitor = 'val_loss',
                           patience = 2)
history = model3.fit(training_padded,
                     y_train,
                     epochs = num_epochs,
                     validation_data = (testing_padded, y_test),
                     callbacks = [early_stop],
                     verbose = 2)

"""**Plotting graph of accuracy of train and validation**"""

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)
# Plotting er code
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.ylim([0, 1])
plt.legend()
plt.grid()
plt.show()

"""**Plotting graph of loss of train and validation**"""

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)
# Plotting er code
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.ylim([0, max(max(train_loss), max(val_loss)) * 1.1])  
plt.legend()
plt.grid()
plt.show()

final_train_accuracy3 = history.history['accuracy'][-1]
final_val_accuracy3 = history.history['val_accuracy'][-1]
print(f"Training Accuracy: {final_train_accuracy:.4f}")
print(f"Validation Accuracy: {final_val_accuracy:.4f}")

models = ['LSTM', 'Bi LSTM', 'GRU']
values = [final_val_accuracy1, final_val_accuracy2, final_val_accuracy3]
plt.bar(models, values, color=['blue', 'green', 'red'])
plt.title('Model Performance Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()
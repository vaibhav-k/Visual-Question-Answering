
# coding: utf-8

# In[17]:


import os
import json
import numpy as np
import cv2 as cv
import pickle
import pandas as pd

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import load_model, Model


# In[2]:


model = load_model('model2.hdf5')


# In[3]:


# model.summary()


# In[4]:


with open('Questions.json') as f:
    que_ans = json.load(f)['questions']

questions = []
images = []
# indices = []

for q in que_ans:
    images.append(q['Image'])
    questions.append(q['Question'])
#     indices.append("Index")

# images = images[:100]


# In[5]:


tokenizer_pkl = open("tokenizer.pkl","rb")
tokenizer = pickle.load(tokenizer_pkl)
tokenizer_pkl.close()

questions_tokenized = tokenizer.texts_to_sequences(questions)
questions_padded = pad_sequences(questions_tokenized, maxlen=40)


# In[6]:


images_processed = []
new_images = []
image_files = os.listdir("./images")

for image in image_files:
    file = os.path.join("./images/", image)
    try:
        image_read = cv.cvtColor(cv.imread(file),cv.COLOR_BGR2RGB)
        image_resized = cv.resize(image_read, (80,60))
        images_processed.append(image_resized)
        new_images.append(image)
    except:
        pass


# In[7]:


images_array = np.array(images_processed, np.float32) / 255.

mean_img = images_array.mean(axis=0)
std_dev = images_array.std(axis=0)
images_normalized = (images_array - mean_img)/ std_dev


# In[9]:


test_images = []
test_questions = []

for idx, val in enumerate(images):
    try:
        u = new_images.index(val + ".png")
        v = images_normalized[u]
        test_images.append(v)
        test_questions.append(questions_padded[idx])
    except:
        pass


# In[14]:


pred = model.predict([test_images, test_questions])


# In[21]:


pred = np.argmax(pred, axis = 1)

correspondence = {
    0: 'brown',
    1: 'metal',
    2: '1',
    3: 'red',
    4: 'purple',
    5: 'gray',
    6: '4',
    7: 'false',
    8: '3',
    9: 'true',
    10: '7',
    11: '6',
    12: '2',
    13: 'rubber',
    14: '0',
    15: 'blue',
    16: 'cylinder',
    17: '5',
    18: 'small',
    19: 'green',
    20: '8',
    21: 'yellow',
    22: 'cyan',
    23: 'cube',
    24: 'large',
    25: 'sphere'
}

pred = [correspondence[i] for i in pred]


# In[25]:


new_df = pd.DataFrame({"Index": [i for i in range(len(pred))], "Answer": pred})
# new_df = pd.DataFrame({"Index": indices, "Answer": pred})
new_df.to_csv("solution.csv", index=False)


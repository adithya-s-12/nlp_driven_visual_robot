#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import json

# Load the dataset
df = pd.read_csv('/home/adithya/commands.csv')

# Create label mappings
unique_objects = df['object'].unique()
object2id = {obj: idx for idx, obj in enumerate(unique_objects)}
id2object = {idx: obj for obj, idx in object2id.items()}

# Apply the label mapping
df['object_id'] = df['object'].map(object2id)

# Split the dataset
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
train_encodings = tokenizer(train_df['command'].tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_df['command'].tolist(), truncation=True, padding=True)

# Convert labels to tensors
train_labels = train_df['object_id'].tolist()
val_labels = val_df['object_id'].tolist()

# Prepare the TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))

# Load the BERT base model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Build the model
input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
token_type_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")

sequence_output = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
cls_token = sequence_output[:, 0, :]
output = Dense(len(unique_objects), activation='softmax')(cls_token)

model = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Convert the datasets to TensorFlow format
train_dataset = train_dataset.shuffle(len(train_df)).batch(16).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# Save the model in the SavedModel format
model.save("bert_model")

# Save the tokenizer and label mappings
tokenizer.save_pretrained('tokenizer')
with open('label_mappings.json', 'w') as f:
    json.dump({'object2id': object2id, 'id2object': id2object}, f)

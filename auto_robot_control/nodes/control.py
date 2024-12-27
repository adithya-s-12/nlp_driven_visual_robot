#!/usr/bin/env python3

import tensorflow as tf
from transformers import BertTokenizer
import json
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('tokenizer')

# Load the label mappings
with open('label_mappings.json', 'r') as f:
    mappings = json.load(f)
    object2id = mappings['object2id']
    id2object = {int(k): v for k, v in mappings['id2object'].items()}

# Load the trained model
model = tf.keras.models.load_model('bert_model')

# Function to classify text into object using the trained model
def classify_text(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    inputs = {k: tf.constant(v) for k, v in inputs.items()}
    inputs['token_type_ids'] = tf.zeros_like(inputs['input_ids'])  # Add token_type_ids
    logits = model(inputs)[0]
    predicted_id = tf.argmax(logits, axis=-1).numpy()[0]
    object_ = id2object[predicted_id]
    return object_

# Function to record audio from the microphone
def record_audio(duration, sample_rate):
    try:
        print("Recording...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        print("Recording complete.")
        sf.write("output.wav", audio, sample_rate)
        return "output.wav"
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None

# Function to convert audio to text
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Audio unintelligible"
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    except Exception as e:
        return f"Error processing audio file: {e}"

if __name__ == '__main__':
    # Choose whether to input text directly or record audio
    mode = input("Type 'text' for text input or 'audio' for audio input: ").strip().lower()
    
    if mode == 'text':
        text_input = input("Enter your command: ")
        object_ = classify_text(text_input)
        print(f"Predicted object: {object_}")
    
    elif mode == 'audio':
        duration = 3  # seconds
        sample_rate = 16000  # Hz
        audio_file = record_audio(duration, sample_rate)
        if audio_file:
            text_from_audio = audio_to_text(audio_file)
            print(f"Recognized text: {text_from_audio}")
            if text_from_audio != "Audio unintelligible":
                object_ = classify_text(text_from_audio)
                print(f"Predicted object: {object_}")
    else:
        print("Invalid mode selected. Please choose either 'text' or 'audio'.")

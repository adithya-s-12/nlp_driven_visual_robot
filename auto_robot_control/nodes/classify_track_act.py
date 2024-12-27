#!/usr/bin/env python3

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load pre-trained BERT model and tokenizer for text classification
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define the classes
action_labels = ["move to"]
object_labels = ["desk-a", "desk-b", "bookshelf-a", "bookshelf-b", "bookshelf-c", "chair", "r-desk"]

# Initialize YOLO model
model = YOLO("/home/adithya/Downloads/best.pt")

# ROS Publishers
annotated_image_pub = rospy.Publisher('/annotated_image', Image, queue_size=10)
camera_info_pub = rospy.Publisher('/camera_info', CameraInfo, queue_size=10)
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

# Function to classify text into action and object using BERT
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Load pre-trained BERT model for action classification
    action_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(action_labels))
    action_outputs = action_model(**inputs)
    action_logits = action_outputs.logits
    predicted_action_id = torch.argmax(action_logits, dim=-1).item()
    action = action_labels[predicted_action_id]

    # Load pre-trained BERT model for object classification
    object_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(object_labels))
    object_outputs = object_model(**inputs)
    object_logits = object_outputs.logits
    predicted_object_id = torch.argmax(object_logits, dim=-1).item()
    object_ = object_labels[predicted_object_id]

    return action, object_

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

# Callback function to process incoming images and check for the desired object
def image_callback(msg, desired_object):
    print(desired_object)
    bridge = CvBridge()
    try:
        # Convert ROS Image message to OpenCV format
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        frame_height, frame_width, _ = frame.shape
        frame_center_x = frame_width / 2
        
        # Perform object detection
        results = model(frame)
        
        # Extract bounding boxes, confidence scores, and class indices
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        # Access names from the first element in results
        names = results[0].names
        # Draw bounding boxes around detected objects and check if desired object is found
        object_found = False
        labels = []
        object_center = None
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            label = names[int(cls)]
            labels.append(label)
            if label == desired_object:
                object_found = True
                object_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                object_height = y2 -y1 # Example distance calculation
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            frame = cv2.putText(frame, f'{label} {score:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)            
        print(labels)
        
        # Publish annotated image
        annotated_image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
        camera_info_pub.publish(CameraInfo())
        print(object_found)
        if object_found:
            rospy.loginfo(f"{desired_object} found. Executing action...")
            move_robot(object_center, object_height, frame_center_x,frame_height)
        else:
            rospy.loginfo(f"{desired_object} not found in the frame.")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    except Exception as e:
        rospy.logerr("Error: {0}".format(e))
# Function to move the robot
def move_robot(object_center, object_height, frame_center_x, frame_height):
    margin = frame_center_x / 5
    desired_height = frame_height * 0.6  # Example: stop when object height is 60% of the frame height

    twist = Twist()
    if object_height < desired_height:
        # Adjust angular velocity to center the object in the camera frame
        if object_center[0] < frame_center_x - margin:  # Object is to the left
            twist.angular.z = -0.3
        elif object_center[0] > frame_center_x + margin:  # Object is to the right
            twist.angular.z = 0.3
        else:
            twist.angular.z = 0.0 
            twist.linear.x = 0.4
    else:
        # Stop the robot
        twist.linear.x = 0.0
        twist.angular.z = 0.0

    cmd_vel_pub.publish(twist)
    rospy.sleep(0.1)

if __name__ == '__main__':
    rospy.init_node('command_processor', anonymous=True)
    
    # Choose whether to input text directly or record audio
    mode = input("Type 'text' for text input or 'audio' for audio input: ").strip().lower()
    
    if mode == 'text':
        text_input = input("Enter your command: ")
        action, object_ = classify_text(text_input)
        rospy.loginfo(f"Predicted action: {action}")
        rospy.loginfo(f"Predicted object: {object_}")
    
    elif mode == 'audio':
        duration = 3  # seconds
        sample_rate = 16000  # Hz
        audio_file = record_audio(duration, sample_rate)
        if audio_file:
            text_from_audio = audio_to_text(audio_file)
            rospy.loginfo(f"Recognized text: {text_from_audio}")
            if text_from_audio != "Audio unintelligible":
                action, object_ = classify_text(text_from_audio)
                rospy.loginfo(f"Predicted action: {action}")
                rospy.loginfo(f"Predicted object: {object_}")
    else:
        rospy.loginfo("Invalid mode selected. Please choose either 'text' or 'audio'.")
    
    if 'object_' in locals() and 'action' in locals() and action == "move to":
        # Subscribe to the ROS camera image_raw topic and pass the desired object
        rospy.Subscriber('/rrbot/camera1/image_raw', Image, image_callback, object_)
        rospy.spin()

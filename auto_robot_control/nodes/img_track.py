#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2

# Initialize YOLO model
model = YOLO("/home/adithya/Downloads/best.pt")

# ROS Publisher for annotated image
annotated_image_pub = rospy.Publisher('/annotated_image', Image, queue_size=10)
camera_info_pub = rospy.Publisher('/camera_info', CameraInfo, queue_size=10)

# Callback function to process incoming images
def image_callback(msg):
    bridge = CvBridge()
    try:
        # Convert ROS Image message to OpenCV format
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Perform object detection
        results = model(frame)
        
        # Extract bounding boxes, confidence scores, and class indices
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        # Access names from the first element in results
        names = results[0].names
        # print(boxes,scores,classes,names)
        # Draw bounding boxes around detected objects
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            label = names[int(cls)]
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            frame = cv2.putText(frame, f'{label} {score:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Publish annotated image
        print(frame)
        annotated_image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
        rospy.loginfo("Annotated image published successfully")
        camera_info_pub.publish(CameraInfo())
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    except Exception as e:
        rospy.logerr("Error: {0}".format(e))

if __name__ == '__main__':
    try:
        # Initialize ROS node
        rospy.init_node('object_detection', anonymous=True)
        rospy.loginfo("ROS node initialized")
        
        # Subscribe to the ROS camera image_raw topic
        rospy.Subscriber('/rrbot/camera1/image_raw', Image, image_callback)
        rospy.loginfo("Subscribed to /rrbot/camera1/image_raw topic")
        
        # Spin to keep the node alive
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


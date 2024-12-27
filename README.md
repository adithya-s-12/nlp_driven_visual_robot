# 1. Getting Gazebo World with Bot:
- Make a workspace and git clone the robot (differential drive bot in this case) into the workspace.
- Now add the camera plugin from `gazebo_ros` plugins.
- Git clone the world (`aws-robomaker-bookstore-world`) in a separate folder.
- Add the contents of the file to `~/.gazebo`.
- Load Gazebo and go to `Insert > Add Path`, search your models folder, and add it.
- Now create the world and save it in your `worlds` folder.
- Go to your launch file.
- Change the `arg world` to `path/to/your/your_world.world`.
- Add `arg world_name` in `include` with value as `$(arg world)`.

### Issues Faced:
- Make sure that the files of all models are present at the correct location and path.
- Ensure to use the correct path and names in the launch file.

---

# 2. Training YOLO or Desired CNN Model to Your Custom Dataset:
- Create a folder of images of the objects to be tracked.
- Upload to Roboflow and annotate them.
- Use Google Colab or Kaggle to train your YOLO model on the custom dataset. Roboflow provides the code to export.
- Paste the code and run the required functions like train, predict, etc.
- Save the model with the best weights in `runs/detect/weights/best.pt`.
- Download the file and load it. Use it to track the objects.

### Issues Faced:
- No major issues were faced.

---

# 3. Using the Model on ROS Camera:
- Subscribe to the topic that is publishing the image.
- Use the `cv_bridge` package to get the video feed.
- Run the model and get the bounding box coordinates, classes, and scores.
- Annotate the image and publish it to a new topic.

### Issues Faced:
- Cannot visualize the new topic in RViz, but can do it using `rqt_image_view`.

---

# 4. Using BERT for NLP:
- Use `speech_recognition`, `sounddevice`, and `soundfile` to convert audio into text.
- Train the BERT model to your custom dataset and use this model to classify the text.
- To train BERT, make a custom dataset of the inputs and the desired outputs and save them in a CSV file.
- Load the file and make sure the classes or outputs are integers. If not, add separate columns with an integer value for each unique label in a class. Multiple classes can have the same integer values.
- Use `scikit-learn` to split the dataset into train and test datasets.
- Use the tokenizer to encode the datasets into a format that BERT can understand.
- Train the BERT model to your custom dataset using TensorFlow.

### Issues Faced:
- **USE TENSORFLOW FOR TRAINING INSTEAD OF PYTORCH.**
- Make sure to convert the labels into IDs and use the IDs, as you cannot directly use the label in string form.

---

# 5. Moving the Robot:
- Write a code to complete the action given.

### Issues Faced:
- Having difficulty getting the distance using only one camera.


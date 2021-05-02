# dog-breed-classifier
Udacity Machine Learning Engineer Capstone Project

#### Definition:
This Capstone Project was a Dog Breed Classifier problem. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. The model implemented was then deployed to a web application using Flask. The datasets were downloaded from the links provided in Step 0 in the jupyter notebook.

##### General Overview of Project Steps:
* Step 0: Import Libraries and Datasets 
* Step 1: Detect Humans using OpenCV’s implementation of Haar feature-based cascade classifier 
* Step 2: Detect Dogs using pre-trained VGG-16 Model (as a starting point) 
* Step 3: Create a CNN to Classify Dog Breeds (from Scratch) 
* Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning) (RESNET-50) 
* Step 5: Write an Algorithm that combines detected humans, detected dogs 
Step 6: Test Your Algorithm on points discussed in Problem Statement 
  * an estimate of the dog’s breed if a dog is detected 
  * an estimate of the dog breed that is most resembling if a human is detected 
  * state not a dog or human if neither a dog or human is detected 
* Step 7: Deploy Model to Web Application using Flask



#### Analysis:
All required files are in this git repo. Unfortunately, the model_transfer.pt could not be uploaded as it was too large. The VGG-16 model in Step 3: Create a CNN to Classify Dog Breeds (from Scratch) provided an accuracy of 14%. In Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning), I decided to use a pre-trained ResNet-50 model. After training (using 75 epochs) and testing the model, it had a test loss of 0.383980 and a test accuracy of 89% (746/836). Having a relatively low test loss and high accuracy score, this was the chosen model. 


A sample output when a human was supplied as an image from the jupyter notebook:

![Sample Output of Human](https://user-images.githubusercontent.com/63424518/116828307-d9459b00-ab6b-11eb-8236-0399f138b67f.png)
![image](https://user-images.githubusercontent.com/63424518/116828713-65f15880-ab6e-11eb-8075-0f7b25e918fd.png)
![image](https://user-images.githubusercontent.com/63424518/116828703-5a9e2d00-ab6e-11eb-8b25-df36ef869822.png)

##### Steps to Run Web App:
* python3 -m venv virtenv (create virtual environement)
* . virtenv/bin/activate (activate virtual environement)
* pip install module (install necessary modules)
* python pip freeze > requirements.txt (save modules that need to be installed)
* pip install -r requirements.txt (install required modules if starting a new session)
* python api.py (run web application)

A sample output when a human was supplied as an image on the web app:
<img width="1170" alt="Sample Output of Dog (web app)" src="https://user-images.githubusercontent.com/63424518/116828276-9be10d80-ab6b-11eb-9c5b-c0e12fde8e79.png">

#### Conclusion
The project model needed to be at least 60% accurate, and I was able to achieve a model with 89% accuracy. I deployed the model to the web app and had a lot of fun doing so and testing it! If you would like to know more about the project, please review Final Report where I elaborate on what I did and why. Overall, I think this capstone project exceeded my learning expectations and further deepended my understanding on Machine Learning. Thank you Udacity for allowing me to take this course!

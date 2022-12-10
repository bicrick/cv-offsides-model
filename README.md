# CV_Offsides_Model
This is a final project for EE379K, Computer Vision class. We are building a computer vision model in python that can detect when someone is offsides in soccer from a still of the game.

Please add the Dataset to the root folder.
Dataset should be structured like this.
    
Dataset/Offside_Images

To run, please create a virtual environment to store all of the python packages.

python -m venv venv

To enter the virtual environment, run

On MacOS:
source ./venv/Scripts/activate

On Windows/Linux
./venv/Scripts/activate

If it is your first time entering the virtual environment, you will need to install the required pip packages. To install all the packages, run:

pip install -r requirements.txt

To freeze your current packages, use:

pip freeze -l > requirements.txt

To run the playerDetection module, please download the yolov3.weights pre-trained model from:

https://pjreddie.com/media/files/yolov3.weights

Place this file in the yolo_cnn folder. Configure the application at the top of main.py. To run, enter the goalDirection and the pictureid from the dataset.
Run main.py. Please note, you may have to change the team designated on offense/defense offsidesCalculation.py.

# BioMed_Final_Project
Source code for final project, By Oren Shapira & Tom Mendel, BME Sarona Oct 16

Project article can be found in the [following link](https://github.com/OrenShapira/BioMed_Final_Project/blob/bdfb9c6d42690c909ac71f2bcbed1d1f03b230b7/Final%20Project%20-%20Tom%20&%20Oren%20-%20%E2%80%8F%E2%80%8FFinal%20submission.pdf)

### Running the project from scratch

1. Clone the project to your local computer

   ```bash
   # Open git bash and run
   cd /local_folder
   git clone https://github.com/OrenShapira/BioMed_Final_Project.git
   ```

2. Install the environment

   1. See introduction in the project article (Section 8)

   2. Visual studio 2015: download from [here](https://my.visualstudio.com/Downloads?q=visual%20studio%202015&wt.mc_id=o~msft~vscom~older-downloads)

   3. Cmake: download from [here]( https://cmake.org/download/)

   4. Anaconda: download anaconda3 5.2.0 from [here](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Windows-x86_64.exe)

      1. Create and activate virtual environment

         ```bash
         # Create virtual environment
         conda create --name opencv-envpython=3.6
         
         # Activate the environment
         conda activate "opencv-envpython=3.6"
         ```

      2. Install OpenCV and supportive packages

         ```bash
         # scikit-learn must be in the same version which create model.joblib in section 4... it is passible to create model.joblib with newer version
         
         # For opencv-python
         pip install opencv-python==3.4.1.15
         
         # For dlib
         conda install -c conda-forge dlib
         ```

3. Running project sections

   For each section, run the .bat file (change file according to your system setup if needed) 

   **Pay attention: not all functionality works because of compatibility issues (newer package versions not fully supported the original project code)** 

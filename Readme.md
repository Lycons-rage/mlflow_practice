# **MLFLOW SETUP STEPS AND NOTES**

### To create a python environment 
```
conda create -p env python=<python_version_here> -y
```

### To activate a created python environment 
```
conda activate <absolute_path_of_env_folder>
```

### Install the requirements after downloading or cloning this repository
```
pip install -r requirements.txt
```

### mlflow GUI command
```
mlflow ui
```

### Benifits of MLFlow:
 - It can handle logging and that too with a nice UI to analyse
 - It can handle logging the result values as well as parameter values
 - It can log even the model and generate the pickle file
 - Earlier we were doing all that manually with code
 - We can download the model pickle file as well as well
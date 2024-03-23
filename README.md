# External agent detector

This project uses some machine learning algorithms to point the presence or absence of an external agent into de controlled environment.
To do so, is necessary collect raw data from a network analizer, manipulate and transform it to make it readable and understandable to the classifiers.
You can have a preview of the script by accessing this *[file](analysis_script.md)*. You'll be able to see analysis outputs and configurations, so that you can know what to expect from the code.

## Getting Started

First of all, create a virtual environment so ensure only necessary packages and file will be running.
You can run the command `python -m venv path/to/project`. After configuration is done, seek for file `activate` in **Scripts** folder and run it with `project\Scripts\activate`.

Your data must be in a folder, classified in **C** and **T** (control and test).
Then, when the code is running, you shoud select the folder where data is in (observe that the window could be behind your active window, in case it does not pop up when you run the script).
This will create a **res** (stands for *results*) folder at data directory where outputs and logs will be saved.
From now on, the script will run by itself, and you can follow up results as them pop up.

### Prerequisites

Run the command `pip install -r requirements.txt` to make sure all packages needed will be installed.


## Running the Tests

After you run the script, a model-like file named `model_classifier.pkl` will be created at **res** folder. 
You can invoke it to test or use it with new data to verify presence of an external agent.
One way you can use to test is is using the `I_am_here_to_test_the_model.ipynb` script to invoke the model and pass the data in the **results_to_submit.csv**. The output shoud be `The prediction is: T`.

## Deployment

Still to come. An executable python file is under construction.


## Authors

* **Alessandro Andreatta, BEng** - *master candidate*
* **Janete E. Zorzi, PhD** - *academic advisor*
* **Cesar Aguzzoli, PhD** - *academic co-advisor*
* **Cláudio Antônio Perottoni, PhD** - *contributor*
* **Luiza Felippi de Lima, MEng** - *contributor*


## License

This project is licensed under GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007

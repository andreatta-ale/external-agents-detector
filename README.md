
# Analix: An external agent detector

This project uses machine learning algorithms to detect the presence or absence of external agents in a controlled environment. It processes raw data from a network analyzer, transforms it, and uses classifiers to detect potential intrusions.

## Table of Contents

- [Analix: An external agent detector](#analix-an-external-agent-detector)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)
  - [Contribution](#contribution)
  - [Authors](#authors)
  - [License](#license)

## Overview

The project aims to identify external agents in networks through data analysis and machine learning. The tool processes input data and uses classifiers to determine if external agents are present.

## Project Structure

- `analysis_script.md`: Preview of the script and its outputs.
- `res/`: Results folder where logs and model outputs are saved.
- `requirements.txt`: List of project dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/andreatta-ale/external-agents-detector.git
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On Linux/Mac: `source venv/bin/activate`
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Organize your data into folders named **C** (control) and **T** (test).
2. Run the main script and select the data folder when prompted.
3. The results will be saved in the `res/` folder.

## Testing

After execution, a model file (`model_classifier.pkl`) will be generated in the `res/` folder. Use the `I_am_here_to_test_the_model.ipynb` script to test the model with new data.

## Contribution

Contributions are welcome! To contribute, follow these steps:
1. Fork the project.
2. Create a new branch (`git checkout -b feature/feature_name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/feature_name`).
5. Open a Pull Request.

## Authors

- **Alessandro Andreatta, BEng** - *Master's Candidate*
- **Janete E. Zorzi, PhD** - *Academic Advisor*
- **Cesar Aguzzoli, PhD** - *Academic Co-Advisor*
- **Cláudio Antônio Perottoni, PhD** - *Contributor*
- **Luiza Felippi de Lima, MEng** - *Contributor*

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.
See the full license text [here](https://www.gnu.org/licenses/gpl-3.0.html).

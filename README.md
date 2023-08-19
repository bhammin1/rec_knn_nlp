# System Project READ ME
## Nearest Neighbor with Semantic Textual Similarity Recommender System
### Brandi Hamming

## Paper
- `bhammin1_system_project_paper.ipynb`

## Code
- Pre-processing
    -  `pre-processing.ipynb`
        -  A notebook that breaks down all of the pre-processing performed on the MIND data
- Recommender System 
    - `news_rec.py`
        - contains all of the functions for pre-processing the MIND data and generating recommendations
    - `experiments1_3.ipynb`
        - contains all of the research experiments 1- 3 (NN, NN with STS and NN with NLI)
    - `experiments_4_5.ipynb`
        - contains experiments 4 and 5 (News only NN and NN with STS)
    - `result_an.ipynb`
        - Contains the evaluation of all experiments
    - `nlp.ipynb`
        - An exploratory notebook for looking into BERT model. Used for the NLP experiments

- Web Application
    - `web_app.py`
        - Run the news recommender application in Streamlit.
        - To start application run the following command in the terminal
            - `streamlit run .\web_app.py `
        - When running the application, you can either create your own account or log in
        - Example Login 
            - userid = `jsmith`
            - password = `abc`
## Configuration
- After pulling the repo, pip install all libraries in `requirement.txt`
- The two config files `configure.yaml` and `configure_new.yaml` are for the web application. No changes are needed in this file
    - New uses added to the system will automatically be added to configure_new.yaml


## Data
- Data Directory
    - contains all MIND data
- Results Directory
    - contains all of the results for research experiment 1-5
# Project1 of Statistical Machine Learning
The main problem in this project is finding edge-relationship in an academic network. Each node in this network represents an author with some information about his/her academic career, and we need judge the existence of a given edge from an initial network.

## Environment

- Windows 10 (Python 3.5+)
- Tensorflow 1.x

## Code

- `Method1_Naive Prediction.py`
- `Method2_Softmax.py`
- `Method3_Filtration Rules.py`
- Just directly run each python file to get the result with corresponding method. Addtionally, in Method3_Filtration.py, you can use `filtration()` function to improve the result gained by the default settings, details can be found in Report.pdf

## Data

- `train.txt` is the training data (raw graph)
- `node.json` contains features of each node in train.txt
- `test-public.csv` is the testing data
- `sample.csv` is an exemplar of form of the expected result
- `pred3-tune.csv` is a develop set, you can generate it with arbitray algorithm then improve result on it with `filtration()` function in Method3_Filtration Rules.py
- More details about the structure of those data can be found in Project1.pdf

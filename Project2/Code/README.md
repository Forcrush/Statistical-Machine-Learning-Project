# Statistical Machine Learning Project2
About transfer learning, more details can be found in `Project2.pdf`.


### Environemnt & Required packages
- Windows 10 (Python 3.5+)
- csv
- numpy
- Tensorflow2
- sklearn
- argparse


### Python File
`baseline.py`: Contains 6 baselines, you can run E`python baseline.py -b 1 -a NN -d MALE`, you can use `-h` instruction for more details.
`FEDA.py`: Feature augmentation method, you can run `python FEDA.py -a NN -d MALE`.
`FSMM.py`: Feature mapping method, you can run `python FSMM.py -d MALE`.
`preprocess.py`: Some basic funtions for data preprocessing.


### Dataset Details
3 domains file: `MALE.csv`, `FEMALE.csv`, `MIXED.csv`.

The dataset comes from the Inner London Education Authority (ILEA), consisting of examination records from 140 secondary schools in years 1985, 1986 and 1987. It is a random 50% sample with 15362 students. The data have been used to study the effectiveness of schools.

The data is sourced from:
    http://www.bristol.ac.uk/cmm/media/migrated/ilea567.zip 

For the purpose of the project, we have processed the dataset as follows:
  - removed the school identifier 
  - removed the gender feature (as this substantially overlaps with school gender)
  - split the dataset by school gender (our "domains")
  - converted the dataset into a more easily readable format


### Feature Coding:
1) Year
```
1985=1; 1986=2; 1987=3
```
2) FSM
```
Percent. students eligible for free school meals
```
3) VR1 Band
```
Percent. students in school in VR band 1
```

4) VR Band of Student
```
VR1=2; VR2=3; VR3=1
```

5) Ethnic group of student
```
ESWI=1*; African=2; Arab=3; Bangladeshi=4; Caribbean=5; Greek=6;Indian=7;Pakistani=8; S.E.Asian=9;Turkish=10; Other=11

*  ESWI: Students born in England, Scotland, Wales or Ireland.
```

6) School denomination
```
Maintained=1; Church of England=2; Roman Catholic=3
```
7) Exam Score
```
Numeric score
```


If you just want to run the experiments you only need to install Python==3.6.8, tensorflow-gpu==1.14.0 ,and Keras==2.2.4
And put glove.6B.300d.txt file in "glove" folder.
And then you can run these three models follow:
Run the demo_intent.py file by typing "python demo_intent.py" in the terminal.
demo_intent.py is used for ID single-task model 
demo_joint_ID_NER.py is used for joint ID and NER model
demo_joint_ID_SF.py is used for joint ID and SF model
The path of the dataset is modified in these three files

If you want to generate a named entities, you need to do the following command
#pip install spacy
#python -m spacy download en_core_web_lg

The purpose of ReturnBig.py file is to convert the first letter of some words in the dataset to uppercase. 
If you don't want to run this step, you can ignore this step. 
The processed data in this step is already in the folder.
And then, run the generate_ner.py, the path of the dataset is modified in file

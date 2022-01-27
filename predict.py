import sys
import spacy

def predict_value(sentence):
    nlp = spacy.load('NERmodel')
    doc = nlp(sentence)
    entity = []
    label = []
    if doc.ents:
        for ent in doc.ents:
            entity.append(ent.text)
            label.append(ent.label_)
        dict = {'entity' : entity  , 'label' : label}
        print(dict)
        return(dict)
    else:
        print('no entities found')

if(len(sys.argv) > 1):
    if sys.argv[1] == '-h':
        print('syntax: python predict -predict "sentence"')
    elif sys.argv[1] == '-predict' and len(sys.argv) == 3:
        sentence = sys.argv[2]
        predict_value(sentence)
    else:
        print("use the proper syntax, 'python -predict.py -predict <sentence>")
else:
    print('use -h for help')
    print('use -predict <"sentence"> to predict value')


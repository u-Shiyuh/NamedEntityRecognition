import spacy
from pathlib import Path
import spacy
import time

from spacy.training import biluo_tags_to_offsets
from spacy.training import iob_to_biluo

start_time = time.time()

nlp = spacy.load("en_core_web_sm")

# read the BILOU file
f = open("tex.bio", "r")

#initialize the variables
sentence = ''
entity = []
TRAIN_DATA = []
words = []
for line in f: 
    entry = line.split()
    if len(entry) == 0:
        items = iob_to_biluo(entity)
        offsets = {'entities' : items }
        created = (sentence.strip(), offsets )
        TRAIN_DATA.append(created)
        #clearing data
        sentence = ""
        sentence = sentence.strip()
        entity = []
    else:
        if sentence == "":
            sentence = entry[1].strip('\n').strip()
        else:
            sentence = sentence + ' ' +  entry[1].strip('\n').strip()
        sentence = sentence.strip()
        entity.append(entry[0])

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
iteration = 0
# Import requirements
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example


# TRAINING THE MODEL
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()

    # Training for 30 iterations
    for iteration in range(500000):
        example = []
    # shuufling examples  before every iteration
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            example = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example.append(Example.from_dict(doc, annotations))
                
                # Update the model
            nlp.update(example, sgd=optimizer, drop=0.2, losses=losses)

        
        output_dir = Path('/NERmodel/')
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.meta['name'] = 'Custom NER model'  # rename model
            nlp.to_disk(output_dir)
            print("--- %s seconds ---" % (time.time() - start_time))
            print("Saved model to", output_dir)
            print(losses)
            iteration += 1
            print(iteration)
            print("------------------------------------------------")

            
        



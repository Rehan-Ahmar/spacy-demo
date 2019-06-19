import spacy
import random

nlp = spacy.load('en')
train_data = [("The startup blew through $1 million", {'entities': [(4, 11, 'ORG')]})]

with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'ner']):
    optimizer = nlp.begin_training()
    for i in range(10):
        random.shuffle(train_data)
        for text, annotations in train_data:
            nlp.update([text], [annotations], sgd=optimizer)
nlp.to_disk('model')
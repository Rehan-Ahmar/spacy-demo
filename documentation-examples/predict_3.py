import spacy

nlp = spacy.load('model')
doc = nlp(u"Apple is looking at buying U.K. startup for $1 billion. ")
for ent in doc.ents:
    print(ent.text, '\t\t', ent.label_) #ent.start_char, ent.end_char
    

import spacy
from spacy.util import minibatch, compounding
import xlrd
import json
import random
from timeit import default_timer as timer

def read_data_from_excel(excel_path):
    wb = xlrd.open_workbook(excel_path)
    train_data = []
    empty_vals = (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK)
    for sheet in wb.sheets():
        for row in range(1, sheet.nrows):
            if (sheet.cell_type(row, 4) in empty_vals) or (sheet.cell_type(row, 5) in empty_vals):
                continue
            sentence = str(sheet.cell(row, 4).value).replace('\n',' ').replace('\r', ' ').strip('\"').lower()
            entities_list = json.loads(str(sheet.cell(row, 5).value).strip('\"'))
            #print(sheet.cell(row, 5).value)
            ner_tag = []
            if len(entities_list) > 0:
                for obj in entities_list:
                    start_end_entity = int(obj.get('start')), int(obj.get('end')), obj.get('entity')
                    if not all(start_end_entity):
                        continue
                    ner_tag.append(start_end_entity)
            train_data.append((sentence, {'entities': ner_tag}))
    print('Number of complete training examples: {}'.format(len(train_data)))
    with open(r'C:\Users\Rehan.Ahmar\Desktop\spacy results\data.json', 'w') as outfile:
        json.dump(train_data, outfile)
    return train_data


def train_ner(train_data):
    nlp = spacy.blank('en')
	
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')

    for raw_text, annotations in train_data:
        '''doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]'''
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(10):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, drop=0.35, sgd=optimizer, losses=losses)
            print("Losses: " + str(losses))
    return nlp


train_data = read_data_from_excel(r'C:\Users\Rehan.Ahmar\Desktop\spacy results\clause_id_128.xlsx')
start = timer()
ner = train_ner(train_data)
end = timer()
print("Training completed in {:0.2f}s".format(end - start))

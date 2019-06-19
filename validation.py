import spacy
from spacy.util import minibatch, compounding
import xlrd
import xlwt
import json
import random
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split

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
    print('Number of examples: {}'.format(len(train_data)))
    with open(r'D:\Demos\spacy-demo\CodeBase\results\data.json', 'w') as outfile:
        json.dump(train_data, outfile)
    return train_data

def validate(nlp, validation_data, epoch_num):
    wb = xlwt.Workbook()
    sheet1 = wb.add_sheet('Sheet1')
    headers = ['Sl. No.', 'Sentence', 'Actual NER tags', 'Detected NERs', 'Status']
    for col_num, item in enumerate(headers):
        sheet1.write(0, col_num, item)
    correct = 0
    for row_num, row in enumerate(validation_data):
        sentence = row[0]
        ner_tags = row[1]['entities']
        doc = nlp(sentence)
        detected_ner_tags = []
        ner_result = []
        for ent in doc.ents:
            start_end_entity = ent.start_char, ent.end_char, ent.label_
            detected_ner_tags.append(start_end_entity)
            ner_result.append((ent.text, ent.start_char, ent.end_char, ent.label_))
        if (sorted(detected_ner_tags) == sorted(ner_tags)):
            correct+=1
            status = 'Success'
        else:
            status = 'Failed'
        sheet1.write(row_num+1, 0, row_num+1)
        sheet1.write(row_num+1, 1, sentence)
        sheet1.write(row_num+1, 2, str(ner_tags))
        sheet1.write(row_num+1, 3, str(ner_result))
        sheet1.write(row_num+1, 4, status)
    wb.save(r'D:\Demos\spacy-demo\CodeBase\validation\validation_epoch_' + str(epoch_num) + r'.xls') 
    print('Validation accuracy after epoch {} : {}'.format(epoch_num, 100 * correct/len(validation_data)))

def train_ner(train_data, validation_data):
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
        '''for itn in range(10):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.01))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, drop=0.2, sgd=optimizer, losses=losses)'''
        for itn in range(10):
            #random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                nlp.update([text], [annotations], drop=0.2, sgd=optimizer, losses=losses)
            print("Losses: {}".format(losses))
            validate(nlp, validation_data, itn+1)
            print('Epoch {} complete.\n'.format(itn+1))
    return nlp


data = read_data_from_excel(r'D:\Demos\spacy-demo\CodeBase\results\Complete data.xlsx')
train_data, validation_data = train_test_split(data, test_size = 0.2, random_state = 0, shuffle=True)
print('Number of training examples: {}'.format(len(train_data)))
print('Number of validation examples: {}'.format(len(validation_data)))
start = timer()
ner = train_ner(train_data, validation_data)
end = timer()
print("Training completed in {:0.2f}s".format(end - start))

import spacy

spacy.info('en')
#spacy.info('en', markdown=True)

nlp = spacy.load('en')
doc = nlp(u"This is a sentence.")
print(doc[0].text)
print(doc[1].text)
print(doc[-1].text)
print(doc[2:3].text)
print([(w.text, w.pos_) for w in doc])

print('*******')
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)
print('*******')	  
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

print('*******')	 
doc = nlp(u'I love coffee')
for word in doc:
    lexeme = doc.vocab[word.text]
    print(lexeme.text, lexeme.orth, lexeme.shape_, lexeme.prefix_, lexeme.suffix_,
          lexeme.is_alpha, lexeme.is_digit, lexeme.is_title, lexeme.lang_)
		  
print(doc.vocab.strings[u'coffee']) # 3197928453018144401
hash_val = doc.vocab.strings[u'coffee']
print(doc.vocab.strings[hash_val])  # 'coffee'
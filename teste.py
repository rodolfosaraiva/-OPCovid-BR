import spacy
nlp_pt = spacy.load('pt_core_news_sm')

doc = nlp_pt('funcionários infectado')


for word in doc:
    print(word.lemma_)


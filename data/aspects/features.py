import numpy as np
import nlpnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse

class AspectExtractor:

	rep_size = 0

	def __init__(self, bow=True, negation=True, emoticon=True, emoji=True, senti_words=True,
						   postag=True, bow_grams=1, verbose=True):
		self.verbose = verbose
		self.bow = bow
		self.negation = negation
		self.emoticon = emoticon
		self.emoji = emoji
		self.senti_words = senti_words
		self.postag = postag

		if verbose: print('using representation: ',end='')

		if senti_words:
			if verbose: print('senti_words ',end='')
			self.make_sentilex()
			self.sentilex_index = self.rep_size
			self.rep_size += 6

		if self.verbose: print()


	def make_bow(self, *args):
		print("")

	def make_sentilex(self, path='./data/aspects/resources/sentilex-reduzido.txt'):
		self.sentilex = {}
		with open(path,'r') as sentilex_file:
			for line in sentilex_file:
				if line[0] != '#':
					line = line.strip().split(',')
					self.sentilex[line[1]] = line[0]

	def get_representation(self, sentences):
		if self.verbose: print(len(sentences))
		if self.verbose: print(self.rep_size)
		myDict = {
			"economia": 0,
			"educação-ciência": 1,
			"localização": 2,
			"pessoa": 3,
			"Político-social": 4,
			"saúde": 5
		}
		new_sentences = np.zeros((len(sentences), self.rep_size))

		for i, sent in enumerate(sentences):
			if self.verbose: print('%i/%i' % (i,len(sentences)),end='\r')

			if self.senti_words:
				sent_words = [0,0,0,0,0,0]
				for sentilexWord in self.sentilex.keys():
					if sentilexWord in sent:
						index = myDict[self.sentilex[sentilexWord]]
						sent_words[index] += 1
				new_sentences[i,self.sentilex_index:self.sentilex_index+6] += np.array(sent_words)

		if self.verbose: print('%i/%i' % (len(sentences),len(sentences)))
		return new_sentences


	def get_representation_tres(self, sentences):
		if self.verbose: print(len(sentences))
		if self.verbose: print(self.rep_size)

		new_sentences = np.zeros((len(sentences), len(self.sentilex)))
		print(new_sentences)
		
		for i, sent in enumerate(sentences):
			if self.verbose: print('%i/%i' % (i,len(sentences)),end='\r')

			if self.senti_words:
				sent_words = [0,0,0,0,0,0]
				for sentilexWord in self.sentilex.keys():
					if sentilexWord in sent:
						index = myDict[self.sentilex[sentilexWord]]
						sent_words[index] += 1
				new_sentences[i,self.sentilex_index:self.sentilex_index+6] += np.array(sent_words)

		if self.verbose: print('%i/%i' % (len(sentences),len(sentences)))
		return new_sentences


	def get_representation_teste(self, sentences):
		myDict = [
			"economia",
			"educação-ciência",
			"localização",
			"pessoa",
			"Político-social",
			"saúde"
		]

		new_sentences = self.get_representation(sentences)

		new_sentences_classified = []
		i = 0

		for sentence in sentences:
			representation = new_sentences[i]

			#sistema de pontos
			j = 0
			for rep in representation:
				if j == 0:
					representation[j] *= 1.1
				if j == 1:
					representation[j] *= 1.1
				if j == 2:
					representation[j] *= 1.1				
				if j == 3:
					representation[j] *= 1.1
				if j == 4:
					representation[j] *= 1.5
				if j == 5:
					representation[j] *= 2

				j += 1

			# Pega número maior
			finalIndex = 5
			j = 0
			for rep in representation:
				if rep > representation[finalIndex]:
					finalIndex = j

				j += 1

			i += 1

			new_sentences_classified.append((sentence, myDict[finalIndex]))

		return new_sentences_classified




	def get_representation_teste_dois(self, sentences):
		myDict = {
			"economia": 0,
			"educação-ciência": 1,
			"localização": 2,
			"pessoa": 3,
			"Político-social": 4,
			"saúde": 5
		}
		new_sentences = []

		for i, sent in enumerate(sentences):
			if self.verbose: print('%i/%i' % (i,len(sentences)),end='\r')
			sent = sent.split()

			if self.senti_words:
				sent_words = [0,0,0,0,0,0]
				for word in sent:
					if word in self.sentilex.keys():
						new_sentences.append((word, self.sentilex[word]))

		return new_sentences
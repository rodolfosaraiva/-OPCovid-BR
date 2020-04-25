import numpy as np

class FeatureExtractor:

	def __init__(self):
		self.make_sentilex()

	def make_sentilex(self, path='./data/sentilex.txt'):
		self.sentilex = {}
		with open(path,'r') as sentilex_file:
			for line in sentilex_file:
				if line[0] != '#':
					line = line.strip().split(',')
					self.sentilex[line[0]] = int(line[1])


	def get_representation(self, sentences):
		new_sentences = []

		for i, sent in enumerate(sentences):
			sent = sent.split()

			for word in sent:
				if word in self.sentilex.keys():
					sentiment = int(self.sentilex[word])
					if sentiment != 0:
						new_sentences.append((word, sentiment))

		return new_sentences
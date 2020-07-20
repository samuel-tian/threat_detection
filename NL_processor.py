import nltk

nltk.download("taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle")

tokens = nltk.word_tokenize("If the boat is chasing and moving slightly erraticaly, then the boat is very likely to be a pirate ship")

print("Parts of Speech: ", nltk.pos_tag(tokens))

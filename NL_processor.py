import nltk

nltk.download("taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle")

#tokens = nltk.word_tokenize("If the boat is chasing and moving slightly erraticaly, then the boat is very likely to be a pirate ship")
tokens = nltk.word_tokenize("The little boy is better than the big girl")


#print("Parts of Speech: ", nltk.pos_tag(tokens))

sentence = nltk.pos_tag(tokens)

grammar = "NP: {<DT>?<JJ>*<NN>}"

cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
result.draw()

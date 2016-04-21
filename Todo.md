* Use ! as an emotion
* Porter Stemming and Lemmatizing available in NLTK - allows you to treat messages, message and messaging the same
* Remove words which occurr in less than ten seasons or something.
* Remove unpopular names
* There are a bunch of unique names 3950 names!!!!
We should filter out unpopular individuals, say all those who have less than X lines in all seasons
* grid search for svm
* Neural Networks, use appropriate activiation functions
* We should somehow plot the datapoints, would be good for the paper too


Best param clf__alpha -> 1e-05
Best param clf__n_iter -> 50
Best param clf__penalty -> elasticnet
Best param tfidf__norm -> l2
Best param tfidf__use_idf -> False
Best param vect__max_df -> 0.5
Best param vect__max_features -> 50000
Best param vect__ngram_range -> (1, 2)

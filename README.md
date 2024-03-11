This is the final project in the IR course, the project attempts to create a search engine for the entire wikipedia corpus.
The following steps were done to create the search engine.
Create an inverted index of the corpus using spark and gcp.
An inverted index was created for the following: 
  title  with stemming
  text with stemming
  title without stemming
  anchor text with steming

We also created the following to improve the search:
  page ranks
  page views

We combined the following methods along with the above after creating the index to improve search accuarcy:
  binary
  tfidf
  bm25
  fast cosine similarity

files:
Search_frontend - a flask app for running the search engine, which calls the search enging initalizer when loading the app that loads the inverted indexes 
se - The core of the search engine, on initializiton created indexes and calculating the idf for each word
queries_train - a json of a search and its results
inverted_index_gcp - a helper with classes for writing reading and saving an inverted index as an object which is pickled.
other files are for tests and the results of these tests.

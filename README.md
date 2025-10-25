#  Questions Project — CS50 AI

This project is part of **Harvard’s CS50’s Introduction to Artificial Intelligence with Python**.  
It implements an **AI-powered question answering system** that uses **TF-IDF** (Term Frequency–Inverse Document Frequency) to find the most relevant sentences that answer a user’s query from a set of text documents.

---

##  Overview

The goal of this project is to build a program that:
- Takes a **query** from the user (like a question).
- Searches through a **corpus of text files**.
- Ranks and returns the **most relevant sentence** or sentences that best match the query.

The program uses **natural language processing (NLP)** techniques — including tokenization, stopword removal, and TF-IDF weighting — to identify important terms and compute how relevant each document and sentence is to the query.

---

##  How It Works

1. **Load Files**  
   Reads all `.txt` files inside a given corpus directory into memory.

2. **Tokenization**  
   Splits text into individual words (tokens), converts them to lowercase, and removes punctuation and stopwords.

3. **Compute IDF (Inverse Document Frequency)**  
   Calculates how important each word is across all documents — rare words get higher IDF scores.

4. **Find Top Files (TF-IDF)**  
   Uses the query words to rank which files are most relevant using TF-IDF scoring.

5. **Extract and Rank Sentences**  
   From the top matching files, splits text into sentences and ranks them by:
   - **Matching Word Measure** (sum of IDF values for query words)
   - **Query Term Density** (fraction of words in the sentence that are query terms)

6. **Return Results**  
   Outputs the best matching sentence(s) that answer the query.

---

##  Concepts Used

* TF-IDF (Term Frequency–Inverse Document Frequency)

* Tokenization

* Stopword Removal

* Information Retrieval

* Sentence Ranking

* Natural Language Processing (NLP)

# This is the CS50 AI Project Questions, which uses TF-IDF (Term Frequency–Inverse Document Frequency) to find the most relevant sentences to a user’s query.

import nltk                                                                     # natural language toolkit for tokenizing and sentence splitting.
import sys                                                                      # handle command-line arguments
import os                                                                       # handle file access
import string                                                                   # used to clean punctuation.
import math                                                                     # used for logarithms in IDF calculation.

# Automatically download required NLTK data (only the first time)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

FILE_MATCHES = 1                                                                # FILE_MATCHES and SENTENCE_MATCHES define how many top results to show.
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])                                             # Load corpus. Reads all text files in the specified folder (e.g. corpus/)
    file_words = {
        filename: tokenize(files[filename])                                     # Tokenize files. Each document is split into a list of lowercase words (no stopwords/punctuation).
        for filename in files
    }
    file_idfs = compute_idfs(file_words)                                        # Compute IDF across corpus. Each word gets a score measuring how rare it is across all files. Common words (like “the”) → low IDF. Rare, informative words (like “evolution”) → high IDF.

    # Prompt user for query
    query = set(tokenize(input("Query: ")))                                     # Ask the user a query. The query is tokenized and turned into a set of words.

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)         # Find top file matches. Finds which file(s) are most relevant to the query using TF-IDF. TF-IDF = Term Frequency × Inverse Document Frequency

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:                                                  # filenames is a list of the top-ranked files (e.g., the top 1 file from the corpus).This loop goes through each of those file names.
        for passage in files[filename].split("\n"):                             # .split("\n") divides the file text wherever there’s a newline character (\n), treating each line or paragraph as a separate passage. This means the code now loops over each paragraph or line in the file.
            for sentence in nltk.sent_tokenize(passage):                        # This line breaks each passage into individual sentences using NLTK’s built-in sentence tokenizer. nltk.sent_tokenize() is smart — it can detect where a sentence ends based on punctuation and capitalization (e.g., . ! ?).
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens                                # Each sentence is stored along with its tokenized words.

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)                                              # Compute IDF for all sentences

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)         # Selects the most relevant sentences (based on IDF and query term density).
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = {}

    # getting the absolute address of the corpus directory
    root = os.path.join(os.getcwd(), directory)

    # It'll hold the list of all files that reside in the data directory
    files_list = os.listdir(root)

    # will go through all the files
    for file_name in files_list:

        # getting the absolute address of the document
        file_dir = os.path.join(root, file_name)

        # lastly, open the doc and save the texts in a dict
        with open(file_dir, mode='r', encoding="utf-8") as fd:
            document = fd.read().rstrip("\n")
            corpus[file_name] = document
    
    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # splitting the doc into words and making lowercase
    tokenized = nltk.word_tokenize(document.lower())

    # removing punctuation, and no English stopwords (like “and”, “the”, “is”)
    punctuation = string.punctuation 
    stopwords = nltk.corpus.stopwords.words('english')

    # it'll hold the cleaned (free from stopwords/punctuations) words only
    cleaned = []

    # will go through each word and filter it out
    for word in tokenized:

        # only consider non-stopwords
        if not word in stopwords:

            # if punctuation is present in the word we need to remove them
            # but strings are immutable, so we need to convert the string into a 'list' of chars
            word_char_list = list(word)

            # go throgh each char of word and check for punctuation
            for i, char in enumerate(word):
                if char in punctuation:
                    word_char_list[i] = ''

            # initializing the word with the new modified list cleaned from punctuations
            modified_word = "".join(word_char_list)

            if len(modified_word) > 0:
                cleaned.append(modified_word)
                

    return cleaned


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Get all words in corpus
    words = set()                                                               # We initialize an empty set called words (sets automatically avoid duplicates).
    for doc_name in documents:
        words.update(documents[doc_name])                                       # words.update(list) adds all words from each document into the set.

    # Calculate IDF's for each word in the corpus
    idfs = dict()                                                               # initialize a dictionary to store IDFs
    for word in words:                                                          # loops over every word in the words set.
        word_appears = sum(word in documents[doc_name] for doc_name in documents) # For each document name in documents, it checks if word is in that document. This creates a generator of True/False values, which sum() turns into a count. After this line, word_appears = number of documents that contain the word.
        idf = math.log(len(documents) / word_appears)                           # Logarithm = (len(documents) = total number of documents N / word_appears = number of documents containing the word nw)
        idfs[word] = idf                                                        # Store it in the dictionary

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Helper function to calculate TF of a word given a doc
    def term_frequency(query_word, doc):                                        # Counts how many times a word appears in a document.
        frequency = 0
        for word in doc:
            if word == query_word:
                frequency += 1
        
        return frequency

    # Calculate TF-IDF first
    tfidfs = dict()
    for file_name in files:                                                     # For each file:
        tfidfs[file_name] = {}                                                  # Create an empty dictionary for its TF-IDF values.
        for word in files[file_name]:                                           # For every word in that file:
            tf = term_frequency(word, files[file_name])                         # Compute TF using term_frequency().
            tfidfs[file_name][word] = tf * idfs[word]                           # Multiply TF by its IDF value. Store that in tfidfs[file_name][word].
    
    # giving each file a score by summing TF-IDF values
    files_score = {}
    for file_name in files:
        files_score[file_name] = 0                                              # Initialize every file’s score as 0.
        for word in query:                                                      # Loop through each query word (the words typed by the user).
            if word in tfidfs[file_name]:                                       # If that query word exists in the file, add its TF-IDF score.
                files_score[file_name] += tfidfs[file_name][word]               # The sum gives the total importance of query words in that file.
    
    # ranking the files based on their scores
    ranked_files = sorted(files_score, key = lambda k: files_score[k], reverse = True) # Sorts the dictionary keys (filenames) by their scores in descending order (highest first).
    
    # will take only the top N files
    top_n = ranked_files[:n]                                                    # Slices the first n filenames from the ranked list (e.g., top 1, top 3, etc.).

    return top_n


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # To hold scores of each sentences
    sentences_score = dict()
    
    # Give each score a 'matching word measure' score and 'query term density' score
    for sentence in sentences:

        # every sentence gonna have 2 kinda score
        score = {'matching word measure': 0, 'query term density': 0}

        # count how many word matches to calculate query term density
        matched_words = 0

        for word in query:                                                      
            if word in sentences[sentence]:                                     # If a query word appears in the sentence, add its IDF weight.
                score['matching word measure'] += idfs[word]                    # High-IDF words are rarer → more important.
                matched_words += 1                                              # matched_words counts how many query words are found in the sentence.

        # calculate query term density    
        score['query term density'] = matched_words / len(sentences[sentence])  # This measures how concentrated query words are within the sentence. If a short sentence contains many query words, it’s likely more relevant.

        # set the calculated score for the sentence to its dictionary
        sentences_score[sentence] = score

    # ranking the sentences based on their matching word measure scores first then query term density scores
    ranked_sentences = sorted(sentences_score, key = lambda k: (sentences_score[k]['matching word measure'], sentences_score[k]['query term density']), reverse = True)

    # will take only the top N sentences
    top_n = ranked_sentences[:n]
    
    
    return top_n


if __name__ == "__main__":
    main()

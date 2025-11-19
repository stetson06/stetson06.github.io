---
layout: post
title: Gauging Alignment with a Base Document Using Natural Language Processing (NLP)
image: "/posts/folders.jpg"
tags: [Deep Learning, Data Viz]

---

# Overview of Project
<br>    
How would you use AI to determine how well certain text aligns with a base document? NLP in Python (Library: scikit-learn)!

The general process is to first convert your base document and the candidate text into numerical representations (called vectors or embeddings), and then use a mathematical formula (like cosine similarity) to calculate how "close" those vectors are to each other. A score close to 1.0 means high alignment, while a score close to 0.0 means low alignment.

I used the Classic Method (TF-IDF + Cosine Similarity) - this method is great for matching keywords and topics but doesn't understand the meaning or context of the words (there are other more advance methods we will discuss later). This Classic Method works by counting how many times important words appear.

How it works:
1. TF-IDF (Term Frequency-Inverse Document Frequency) creates a vector for each document, where each dimension is a word. The value is high if a word is frequent in that document but rare in all documents.
2. Cosine Similarity calculates the angle between these two vectors.

<br>                                                                      
```python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_doc = "This is the main document about finance and economics."
text_to_compare = "A new article discusses economic trends and financial markets."
unrelated_text = "The quick brown fox jumps over the lazy dog."

documents = [base_doc, text_to_compare, unrelated_text]

# 1. Create the TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 2. Calculate the similarity
# This compares the first doc (index 0) to all other docs
scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

print(f"Score for aligned text: {scores[0][1]}")   # e.g., 0.45
print(f"Score for unrelated text: {scores[0][2]}") # e.g., 0.0

```
<br>

Best for: Document-sorting, information retrieval, and finding documents with similar keywords.















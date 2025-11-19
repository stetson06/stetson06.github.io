---
layout: post
title: Gauging Alignment with a Base Document Using Natural Language Processing (NLP)
image: "/posts/folders.jpg"
tags: [Deep Learning, Data Viz]

---

# Overview of Project

One time, a customer with a large R&D budget wanted some help identifying the latest technologies for investment, prioritizing them in relation to the overall organization's strategic plan and long-term goals and objectives. In effect, they asked us: "How would you use AI to determine how well certain text (descriptions of candidate technologies) aligns with a base document (strategic plan document)?" We soon determined the best analytical methodology that was available at the time - Natural Language Processing (NLP) in Python using the scikit-learn library! There are now other, more advanced methods available that we will discuss at the end of this page.

<br>  
# NLP Technique
The general process is to first convert your base document and the candidate text into numerical representations (called vectors or embeddings), and then use a mathematical formula (like cosine similarity) to calculate how "close" those vectors are to each other. A score close to 1.0 means high alignment, while a score close to 0.0 means low alignment.

We used the Classic Method (TF-IDF + Cosine Similarity) - this method is great for matching keywords and topics but doesn't understand the meaning or context of the words. This Classic Method works by counting how many times important words appear.

How it works:
1. TF-IDF (Term Frequency-Inverse Document Frequency) creates a vector for each document, where each dimension is a word. The value is high if a word is frequent in that document but rare in all documents.
2. Cosine Similarity calculates the angle between these two vectors (with a lower angle indicating directional "similarity" and semantic alignment).

We first looked up the organization's current strategic planning document and loaded it onto a Word (.docx) file. We then obtained Gartner's latest Emerging Technolgies Hype Cycle publication (2021) that listed 34 ETs across three themes - see below:

Theme 1: Engineering Trust (x10)
•	Sovereign Cloud
•	Nonfungible Tokens (NFTs)
•	Machine-Readable Legislation
•	Decentralized Identity (DCI)
•	Decentralized Finance (DeFi)
•	Homomorphic Encryption
•	Active Metadata Management
•	Data Fabric
•	Real-Time Incident Center
•	Employee Communications Applications
Theme 2: Accelerating Growth (x6)
•	Generative AI
•	Digital Humans
•	Multi-experience
•	Industry Cloud
•	AI-Driven Innovation
•	Quantum Machine Learning (Quantum ML)
Theme 3: Sculpting Change (x18)
•	Composable Applications
•	Composable Networks
•	AI-Augmented Software Engineering
•	AI-Augmented Design
•	Physics-Informed AI
•	Influence Engineering
•	Digital Platform Conductor Tools
•	Named Data Networking (NDN)

Each ET had a description that wsa captured in a separate Word file. An example for one of the ETs, "Generative AI", is captured below:
Generative AI: AI techniques that learn from existing artifacts to generate new, realistic content (images, text, audio, code) that reflects the characteristics of the training data but does not repeat it. (In 2021, this was just beginning to rise significantly).

We posted all the Word files onto our working repository, ready for Python to come calling!

To code this all up, we first had to install the required library:

<br>                                                                      
```
python -m pip install python-docx

```
<br>                                                                      
Run this command in your Terminal (Mac/Linux) or Command Prompt/PowerShell (Windows) ("C:\Users\YOURACCOUNTNAME>") - if running Python through Anaconda, then use the Anaconda prompt and the command below:

```
conda install -c conda-forge python-docx

```
<br>  
This is the Anaconda native way and is recommended. If this doesn't work (takes too long or fails), then use this #inside this same window#:

```
pip install python-docx

```
<br>  

Once the installation finishes, you may now run the Python script as normal. Whatever you do, do ##not## run this preliminart step inside the Python interpreter (the place with the >>> prompt), or you will get a syntax error.

Then we proceded with our Python code as below:

```python
import os
import glob
from docx import Document

def extract_text_from_docx(file_path):
    """
    Helper function to open a .docx file and join its paragraphs into a single string.
    """
    try:
        doc = Document(file_path)
        full_text = []
        
        # Iterate through paragraphs and append text
        for para in doc.paragraphs:
            full_text.append(para.text)
            
        # Join all paragraphs with a newline character
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading file: {e}"

def read_all_docx_in_folder(folder_path):
    """
    Scans a folder for .docx files and returns a dictionary.
    Structure: {'filename': 'file content string'}
    """
    # Create a search pattern for .docx files
    search_pattern = os.path.join(folder_path, "*.docx")
    files = glob.glob(search_pattern)
    
    file_contents = {}

    if not files:
        print("No .docx files found in the specified directory.")
        return file_contents

    print(f"Found {len(files)} files. Processing...")

    for file_path in files:
        # Get just the filename (e.g., "resume.docx") for the key
        file_name = os.path.basename(file_path)
        
        # Extract text
        text_content = extract_text_from_docx(file_path)
        
        # Store in dictionary
        file_contents[file_name] = text_content

    return file_contents

# --- Usage ---
# Use '.' for the current directory, or specify a full path like 'C:/Documents/Reports'
target_directory = '.' 

all_documents = read_all_docx_in_folder(target_directory)

# Example: Print the first 100 characters of each file
for name, content in all_documents.items():
    print(f"--- {name} ---")
    print(content[:100] + "...")
    print("\n")

```

<br>

Important Limitation (Tables)
The standard doc.paragraphs loop shown above reads the body text. It often skips text inside tables. So if your documents rely heavily on tables, you must also iterate through doc.tables. Here is how you would modify the extract_text_from_docx function to include table text:

```python
def extract_text_including_tables(file_path):
    doc = Document(file_path)
    full_text = []

    # Extract paragraph text
    for para in doc.paragraphs:
        full_text.append(para.text)

    # Extract table text
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                full_text.append(cell.text)
                
    return '\n'.join(full_text)

```

<br>  



```python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


base_doc = ""
text_to_compare = ""
unrelated_text = ""

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

This method is best for: Document-sorting, information retrieval, and finding documents with similar keywords.

<br>

https://www.zdnet.com/article/gartner-releases-its-2021-emerging-tech-hype-cycle-heres-whats-in-and-headed-out/

# tech1 = Nonfungible Tokens (NFTs)
# tech2 = Active Metadata Management
# tech3 = Generative AI
# tech4 = AI-Driven Innovation
# tech5 = Quantum Machine Learning (Quantum ML)



















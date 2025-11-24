---
layout: post
title: Gauging Alignment with a Base Document Using Natural Language Processing (NLP)
image: "/posts/folders.jpg"
tags: [Deep Learning, Data Viz]

---

# Overview of Project

A customer with a large R&D budget wanted some help identifying the latest technologies for investment as well as prioritizing them in relation to their organization's strategic vision and plan. In effect, they asked us: **"How would you use AI to determine how well certain text (i.e., descriptions of candidate technologies) aligns with a target document (i.e., strategic plan document)?"** We soon determined the best analytical methodology would be to leverage *Natural Language Processing (NLP)* in Python using the scikit-learn library or an *Artificial Neual Network*! We discuss both in this case study.

The general process, common to both, is to first convert the target/base document and the candidate text into numerical representations (called *vectors* or *embeddings*), and then use a mathematical formula (like *cosine similarity*) to calculate how "close" those vectors are to each other. A score close to 1.0 means high alignment, while a score close to 0.0 means low alignment.

<br>  
# Old School NLP Technique

We first will show how to use the Classic Method (*TF-IDF + Cosine Similarity*) - this method is great for matching keywords and topics but doesn't understand the meaning or context of the words. This Classic Method works by counting how many times important words appear. Before the age of AI and Deep Learning, this was the industry standard for text comparison. 

This method treats documents as a "bags of words." It calculates alignment based on keyword overlap, statistically weighing unique words (like "Synergy" or "Q3-Revenue") higher than common words (like "the" or "is").

How it works:
1. *TF-IDF* (*Term Frequency-Inverse Document Frequency*) creates a vector for each document, where each dimension is a word. The value is high if a word is frequent in that document but rare in all documents.
2. *Cosine Similarity* calculates the angle between these two vectors (with a lower angle indicating directional "similarity" and semantic alignment).

We first looked up the organization's current vision/strategic planning document and loaded it onto a Word (.docx) file. We subsequently obtained Gartner's then-current 2021 **Emerging Technolgies Hype Cycle** [publication](https://www.zdnet.com/article/gartner-releases-its-2021-emerging-tech-hype-cycle-heres-whats-in-and-headed-out/) that listed 24 ETs across three themes - see below:

| **Theme 1: Engineering Trust** | **Theme 2: Accelerating Growth** | **Theme 3: Sculpting Change** |
|:---|:---|:---|
|Sovereign Cloud|Generative AI|Composable Applications|
|Nonfungible Tokens (NFTs)|Digital Humans|Composable Networks|
|Machine-Readable Legislation|Multi-experience|AI-Augmented Software Engineering|
|Decentralized Identity (DCI)|Industry Cloud|AI-Augmented Design|
|Decentralized Finance (DeFi)|AI-Driven Innovation|Physics-Informed AI|
|Homomorphic Encryption|Quantum Machine Learning (Quantum ML)|Influence Engineering|
|Active Metadata Management||Digital Platform Conductor Tools|
|Data Fabric| |Named Data Networking (NDN)|
|Real-Time Incident Center| | |
|Employee Communications Applications| | |

<br>

When these ETs are plotted out across the near future (based on expected arrival of the technology), it looked like this:
<br>
    ![gartner](/img/posts/gartner.png)

<br>

Each ET had a description that was captured in a separate Word file. An example for one of the ETs, "Generative AI", is captured below:
<br>
**Generative AI: AI techniques that learn from existing artifacts to generate new, realistic content (images, text, audio, code) that reflects the characteristics of the training data but does not repeat it.** (In 2021, this was just beginning to rise significantly).

We posted all the Word files onto our working repository, ready for Python to come calling!

To code this all up, we first had to install the required library:
```
python -m pip install python-docx
```                                                                     
We ran this command above in our Terminal (Mac/Linux) or Command Prompt/PowerShell (Windows) ("C:\Users\YOURACCOUNTNAME>") - if running Python through Anaconda, then use the Anaconda prompt and the command below:

```
conda install -c conda-forge python-docx
```
<br>  
This is the Anaconda native way and is recommended. If this doesn't work (takes too long or fails), then use this *inside this same window*:

```
pip install python-docx
```
<br>  

Once the installation finished, we then were poised to run the Python script as normal. 

Since the work we performed for our client was proprietary and/or classified, for this case study we will use a notional customer - *GitHub, Inc.* - and use their current vision statement found online as the target document.

Also, for simplicity's sake, we will only involve five of the 24 emerging technology candidates in the NLP model, as listed below (for a total of six Word .docx files):

|Candidate Name |Emerging Technology |
|:---|:---|
|tech1|Nonfungible Tokens (NFTs)|
|tech2|Active Metadata Management |
|tech3|Generative AI|
|tech4|AI-Driven Innovation|
|tech5|Quantum Machine Learning (Quantum ML)|

<br>
Just eyeballing the ETs, and given GitHub's mission/vision (i.e., *"To build the AI-native developer platform for the world."*), we would guess that tech3 (Generative AI) and and tech4 (AI-Driven Innovation) would have the highest alignment with GitHub, while tech1 (NFTs) would have the lowest. Let's see what happens!

Our Python code begins as below to extract the text from all six Word files in our working directory (and any other .docx files therein!) and capture them as strings in a combined dictionary (a single object).

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
<br>
The standard doc.paragraphs loop shown above reads the body text. It often skips text inside tables. So if our documents had relied heavily on tables (which they didn't), then we also would have had to iterate through doc.tables. Here is how we would haved modified the extract_text_from_docx function to include table text:

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
Below is what the output looks like:
    ![strings](/img/posts/strings.png)
    
<br>
This is when the fun really starts! We see below Python's scikit-learn library, the Vectorizer function, and cosine_similarity calculator at work in the code. 

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def calculate_tfidf_alignment(documents_dict, target_filename):
    # --- 1. Preparation ---
    # Convert dictionary to parallel lists for Scikit-Learn
    filenames = list(documents_dict.keys())
    texts = list(documents_dict.values())

    if target_filename not in filenames:
        return f"Error: '{target_filename}' not found."

    # --- 2. Vectorization (TF-IDF) ---
    # This converts text into a matrix of numbers based on word counts.
    # stop_words='english' removes common filler words (the, a, an, in)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the texts. 
    # This creates a sparse matrix where rows=documents, columns=words
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # --- 3. Calculate Similarity ---
    # Find the index location of our target file in the list
    target_index = filenames.index(target_filename)
    
    # Calculate cosine similarity between the target and ALL other documents
    # This returns an array of scores (0.0 to 1.0)
    cosine_scores = cosine_similarity(tfidf_matrix[target_index:target_index+1], tfidf_matrix)

    # --- 4. Format Output ---
    results = []
    
    # The result is a list of lists, we want the first (and only) row
    scores_list = cosine_scores[0]

    for i, score in enumerate(scores_list):
        # Compare against others, not itself
        if filenames[i] != target_filename:
            results.append({
                "Document": filenames[i],
                "Keyword Overlap Score": score,
                "Percentage": f"{score:.1%}"
            })

    # Sort by highest score
    results_df = pd.DataFrame(results).sort_values(by="Keyword Overlap Score", ascending=False)
    
    return results_df

# --- Usage ---
target_file = "strategicplanning.docx"

tfidf_report = calculate_tfidf_alignment(all_documents, target_file)

print(f"Keyword Alignment with {target_file}:\n")
print(tfidf_report)

```

The output is as below:
<br>
    ![angles](/img/posts/angles.png)
<br>
It should come as no surprise that tech3 (Generative AI) fared well, but tech4 (AI-Driven Innovation) only came in a distant second (given GitHub's emphasis on innovation, it would seem it should have fared better). Not doing so well were tech1 (NFTs) and tech5 (Quantum ML) - tech1 was expected, but tech5 was admittedly a minor surprise (either that or we don't really know what quantum ML is)! 

<br>
<br>
# State-of-the-Art Deep Learning/Artificial Neural Network Technique
To conduct "semantic alignment" (meaning checking how closely the meaning of documents matches, rather than just checking for matching keywords), the industry standard approach currently is to use *Vector Embeddings*.

We use the *sentence-transformers library* (based on **BERT** or *bidirectional encoder representations from transformers*). It converts every document into a list of numbers (a vector) representing its meaning. As before, we then calculate the Cosine Similarity between the vectors to get a score from 0 (no alignment) to 1 (perfect alignment).

We first install the required Python Sentence Transformers library using either the Windows Command or Anaconda prompt (as before):

```python
conda install -c conda-forge sentence-transformers
```

<br>
The ensuing Python code block is captured below:

```python
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# 1. Load a pre-trained model
# 'all-MiniLM-L6-v2' is excellent for speed and accuracy on local CPUs
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_semantic_alignment(documents_dict, target_filename):
    # --- Validation ---
    if target_filename not in documents_dict:
        return f"Error: '{target_filename}' not found in the document list."
    
    # --- Segregate Data ---
    # Isolate the text of the target strategy document
    target_text = documents_dict[target_filename]
    
    # Create lists for the other files to compare against
    other_filenames = []
    other_texts = []
    
    for filename, text in documents_dict.items():
        if filename != target_filename:
            other_filenames.append(filename)
            other_texts.append(text)
            
    if not other_texts:
        return "No other documents to compare against."

    print("Generating embeddings... (this uses your CPU to 'read' the texts)")
    
    # --- Encoding (The Magic Step) ---
    # Convert the target text to a vector
    target_embedding = model.encode(target_text, convert_to_tensor=True)
    
    # Convert all other texts to vectors (in a batch)
    corpus_embeddings = model.encode(other_texts, convert_to_tensor=True)

    # --- Calculate Cosine Similarity ---
    # Computes how close the vectors are in space
    # returns a tensor of scores
    cosine_scores = util.cos_sim(target_embedding, corpus_embeddings)[0]

    # --- Format Results ---
    results = []
    for i in range(len(other_filenames)):
        score = cosine_scores[i].item() # Convert tensor to float
        results.append({
            "Document": other_filenames[i],
            "Alignment Score": score,
            "Percentage": f"{score:.1%}"
        })

    # Sort by highest score first
    results_df = pd.DataFrame(results).sort_values(by="Alignment Score", ascending=False)
    
    return results_df

# --- Usage ---
# Assuming 'all_documents' is the dictionary from the previous code block:

target_file = "strategicplanning.docx"

# Run the analysis
alignment_report = calculate_semantic_alignment(all_documents, target_file)

# Print the table
print(f"Semantic Alignment with {target_file}:\n")
print(alignment_report)
```

<br>
### How to Interpret the Output
The code outputs a Pandas DataFrame (a nice table). Here is what the Alignment Score means:

| **Score** | **Alignment** | **Description** |
|:---:|:---|:---|
|0.80 - 1.00|Highly Aligned|The documents are likely talking about the exact same topics, perhaps even reusing the same paragraphs.|
|0.50 - 0.79|Moderately Aligned|They share the same context (e.g., both are about "Corporate Strategy" or "Q3 Goals"), but the specific content differs.|
| 0.20 - 0.49|Loosely Related|They might share a domain (e.g., "Business"), but the topics don't overlap much.|
|< 0.20|Unrelated|One is about Strategy, the other is likely about something completely different, like a lunch menu or unrelated IT logs.|

#### Important Note on Document Length
The model used here (all-MiniLM-L6-v2) typically looks at the first 256–512 "tokens" (roughly 300-400 words) to form its impression of the document.

If your documents are short: This works perfectly.
If your documents are 50+ pages: It will only compare the introductions/summaries.

The output now is as below:
<br>
    ![angles2](/img/posts/angles2.png)
<br>
The Pareto chart graphically captures the relative performance of the five ETs involved in this case study.
<br>
    ![pareto](/img/posts/pareto.png)
<br>

Very interesting! Notice how tech4 (AI-Driven Innovation) is now first, by a good margin over the previous top ET (tech3 - Generative AI), better reflecting the ethos at GitHub. The rest of the candidates remain in the same order, although it's surprising that tech5 (Quantum ML) still does so poorly.

<br>
So **which is the better method**, TF-IDF (old) or BERT (new)? They are compared in the table below:
<br>
    ![oldvsnew](/img/posts/oldvsnew.png)

Generally speaking, if we want to know if other documents reference the specific jargon used in target document, then use TF-IDF; but if we want to know if other documents support the goals of the target document (even if they describe them differently), use BERT, which reads for **meaning and context**.

<br>
# Growth & Next Steps

Consulting current LLMs may prove fruitful in shedding light on more efficient ways to conduct this project using Sentence Transformers (the current "cutting-edge" technique). Advancements in Generative AI promise to provide even more advanced techniques to gauge semantic alignment in the coming years - perhaps with more processing power, there will be more "depth" to the "deep learning" in terms of increased # of parameters involved for better model training or even a paradigm shift that a brilliant programmer will devise that will shift the landscape on what is considered the "best" or new "cutting-edge" methodology. So let's be on the lookout for this - the future is bright!

___









































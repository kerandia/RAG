# Logistics Feedback Analysis using Retrieval-Augmented Generation (RAG)
I initially used a Jupyter Notebook (rag1.ipynb) for trying and testing my code and the RAG system. This allowed me to iteratively develop and refine my approach before creating the final Python script (logisticrag.py).

## Limitations
The RAG system may not achieve 100% success in all cases due to the inherent limitations of the models and the quality of the feedback data.
The system relies on the accuracy of the embedding and language models, which may not always perfectly capture the nuances of the feedback data.

## Overview
This project demonstrates the implementation of a RAG system to analyze customer feedback for a logistics company. While the system provides a powerful approach to generating relevant responses, it is important to acknowledge its limitations and continuously improve the models and data quality for better performance.


## Assumptions
- The feedback data is stored in a JSON file (`feedback_data.json`) and is structured as a list of dictionaries, each containing a `feedback` key.
- The embedding model (`hf.co/CompendiumLabs/bge-base-en-v1.5-gguf`) and language model (`hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`) from `ollama` are used to generate embeddings and responses.
- The system retrieves the top N most relevant feedback chunks based on cosine similarity between the query and feedback embeddings.

## Analysis
- **Data Loading**: The feedback data is loaded from `feedback_data.json` and each feedback entry is converted into an embedding vector using the embedding model.
- **Vector Database**: A vector database is maintained to store feedback chunks and their corresponding embedding vectors.
- **Retrieval**: The retrieval function calculates the cosine similarity between the query embedding and feedback embeddings to find the most relevant feedback chunks.
- **Generation**: The retrieved feedback chunks are used to construct a prompt for the language model, which generates a response based on the provided context.

## Key Decisions
- **Embedding Model**: The `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` embedding model was chosen for its effectiveness in generating meaningful embeddings for text data.
- **Language Model**: The `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` language model was selected for its ability to generate coherent and contextually relevant responses.
- **Cosine Similarity**: Cosine similarity was used as the metric to measure the relevance of feedback chunks to the query, as it effectively captures the semantic similarity between vectors.
- **RAG System**: The RAG system was chosen for its ability to combine retrieval-based and generation-based approaches, providing a balance between accuracy and relevance. However, it is acknowledged that the system may not achieve 100% success in all cases.

## Usage
1. **Install Dependencies**: Ensure that the required libraries (`ollama`, `concurrent.futures`, `json`, `os`, `time`) are installed.
2. **Prepare Feedback Data**: Ensure that the `feedback_data.json` file is present and correctly structured.
3. **Run the Script**: Execute the `logisticrag.py` script to start the system. The user can interact with the system by asking questions, and the system will provide responses based on the customer feedback data.

## Example
```sh
python logisticrag.py
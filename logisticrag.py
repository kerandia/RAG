
import os

os.system('ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf')
os.system('ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF')

os.system('pip install ollama')


import concurrent.futures
import time
import json
import ollama

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
      # temporary list to store (chunk, similarity) pairs
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    # sort by similarity in descending order, because higher similarity means more relevant chunks
    similarities.sort(key=lambda x: x[1], reverse=True)
     # finally, return the top N most relevant chunks
    return similarities[:top_n]
#defined instruction prompt regarding the tasks to be performed by the AI agent
def chat_with_ollama(input_query, formatted_chunks):
    instruction_prompt = f"""You are an AI agent. Use only the following customer feedback to generate a summary. Ensure accuracy, relevance, and consistency:{formatted_chunks}"""

    print(f"Instruction Prompt: {instruction_prompt}")

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )
#defined the chatbot response to the user query
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

def main():
    # Load feedback data and add to vector database
    feedback_file = 'feedback_data.json'
    if not os.path.exists(feedback_file):
        print("Feedback file not found.")
        return

    with open(feedback_file, 'r', encoding='utf-8') as f:
        feedback_data = json.load(f)

    customer_feedback = feedback_data.get('customer_feedback', [])
    for i, feedback in enumerate(customer_feedback):
        if isinstance(feedback, dict) and 'feedback' in feedback:
            add_chunk_to_database(feedback['feedback'])
            print(f'Added chunk {i+1}/{len(customer_feedback)} to the database')

    while True:
        input_query = input('Ask me a question (or type "exit" to quit): ')
        if input_query.lower() in ['exit', 'quit']:
            print("Ending conversation.")
            break

        retrieved_knowledge = retrieve(input_query, top_n=3)

        if not retrieved_knowledge:
            print("No relevant feedback found.")
            continue

        print('Retrieved knowledge:')
        for chunk, similarity in retrieved_knowledge:
            print(f' - (similarity: {similarity:.2f}) {chunk}')

        formatted_chunks = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
        print(f"Formatted Chunks: {formatted_chunks}")
        chat_with_ollama(input_query, formatted_chunks)

if __name__ == "__main__":
    main()

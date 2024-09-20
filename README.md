# Retrieval-Augmented Generation (RAG) QA Bot

## Overview

The Retrieval-Augmented Generation (RAG) QA Bot is a machine learning application that combines document retrieval and generative AI to answer questions based on uploaded PDF documents. The bot utilizes a vector database (Pinecone) for efficient retrieval of relevant information and employs OpenAI and Cohere models for generating coherent answers.

## Features

- Upload PDF documents for analysis.
- Ask questions related to the content of the uploaded documents.
- Retrieve relevant document segments alongside generated answers.
- Select between OpenAI and Cohere APIs for embeddings.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example Queries](#example-queries)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributions](#contributions)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Requirements

- Python 3.7 or higher
- Required Python packages:
  - `streamlit`
  - `langchain`
  - `pinecone-client`
  - `python-dotenv`
- Access to OpenAI and/or Cohere API keys.
- A Pinecone account for vector database access.

## Installation

1. **Clone the repository:**

   git clone https://github.com/jindalayush326/RAG-QA-bot.git
   cd RAG-QA-bot
   
2. **Set up a virtual environment (optional but recommended):**

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install the required packages:**
   
Pinecone
Streamlit

4. **Create a .env file in the project root and add your API keys:**

OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key

## Usage
1. **Run the Streamlit app:**
    
streamlit run app.py

2. **Interact with the app:**

Upload a PDF document using the file uploader.
Type your question in the input box and press Enter.

3. **View results:**

The application will display the generated answer and the relevant document segments.

## Example Queries

"What is the main topic of the document?"
"Summarize the key findings."
"What recommendations are provided in the text?"

## Documentation
Model Architecture: This RAG model combines retrieval and generative components. It retrieves relevant segments from Pinecone based on the user query and generates answers using a language model.
Retrieval Process: Document chunks are embedded and stored in a Pinecone vector database. Upon receiving a query, similar embeddings are retrieved through similarity search.
Generative Responses: The retrieved document segments are processed by the selected generative model (OpenAI or Cohere) to create coherent answers.

## Troubleshooting
Ensure that your API keys are valid and have the necessary permissions.
If you encounter issues with PDF uploads, verify that the file format is supported and not corrupted.

## Contributions
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Langchain for providing foundational libraries.
Pinecone for offering the vector database solution.
OpenAI and Cohere for the powerful generative models.

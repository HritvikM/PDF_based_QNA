# Import necessary libraries
from PyPDF2 import PdfReader  # PDF parsing
from faiss import IndexFlatL2  # Vector store
from sentence_transformers import SentenceTransformer  # Text embedding
from transformers import pipeline  # Question answering

# Define functions
def process_pdf(pdf_path):
  """
  Parses a PDF file and returns the extracted text.
  """
  if pdf_path is not None:
    # Open the PDF file using PdfReader
    pdf_reader = PdfReader(pdf_path)

    # Initialize an empty string to store extracted text
    text = ""

    # Loop through each page in the PDF
    for page in pdf_reader.pages:
      # Extract the text from the current page and append it to the text variable
      text += page.extract_text()

    # Return the extracted text
    return text


def embed_text(text):
  """
  Embeds text into vector representations.
  """
  model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
  return model.encode(text)

def build_vector_store(texts):
  """
  Builds a vector store for efficient search.
  """
  vectors = [embed_text(text) for text in texts]
  index = IndexFlatL2(len(vectors[0]))
  index.add(vectors)
  return index

def query_vector_store(query, index, texts):
  """
  Searches the vector store for relevant documents based on a query.
  """
  query_embedding = embed_text(query)
  distances, indices = index.search(query_embedding.reshape(1, -1), k=5)
  return [texts[i] for i in indices[0]]

def answer_question(question, passage):
  """
  Extracts an answer from a passage based on a question.
  """
  qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
  answer = qa_model(question=question, context=passage)
  return answer["answer"]

# Define chatbot logic
def chat(pdf_path):
  """
  Runs the chatbot interaction loop.
  """
  # Process PDF and build vector store
  text = process_pdf(pdf_path)
  sentences = text.strip().split("\n")
  vector_store = build_vector_store(sentences)

  # Chat loop
  while True:
    user_query = input("Ask your question: ")
    if user_query == "quit":
      break

    relevant_sentences = query_vector_store(user_query, vector_store, sentences)
    for sentence in relevant_sentences:
      answer = answer_question(user_query, sentence)
      print(f"\nPossible Answer: {answer}")

    print("\n")

# Run the chatbot
chat("222.pdf")

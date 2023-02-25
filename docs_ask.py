"""Ask a question to the notion database."""
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse

parser = argparse.ArgumentParser(description="Ask a question to the notion DB.")
parser.add_argument("question", type=str, help="The question to ask the notion DB")
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("awsdocs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)


prompt_template = """You are a very enthusiastic AWS solutions architect who loves to help people! Given the following sections from the AWS documentation, answer the question using only that information, outputted in markdown format. If you are unsure and the answer is not explicitly written in the documentation, say "Sorry, I don't know how to help with that."

Context sections:
{context}

Question: {question}

Answer as markdown (inluding related code snippets if available):
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(
    llm=OpenAI(temperature=0), vectorstore=store
)
result = chain({"question": args.question})
print()
print()

print(f"Question: {args.question}")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")

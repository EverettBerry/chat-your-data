"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import re
import sys


class AWSMarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Markdown-formatted headings."""

    def __init__(self, **kwargs):
        """Initialize a MarkdownTextSplitter."""
        separators = [
            # First, try to split along Markdown headings
            "## ",
            "### ",
            "#### ",
            "##### ",
            "###### ",
            # Note the alternative syntax for headings (below) is not handled here
            # Heading level 2
            # ---------------
            # End of code block
            "```\n\n",
            # Horizontal lines
            "\n\n***\n\n",
            "\n\n---\n\n",
            "\n\n___\n\n",
            # Note that this splitter doesn't handle horizontal lines defined
            # by *three or more* of ***, ---, or ___, but this is not handled
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)


def clean_doc(d):
    remove_html = re.compile("<.*?>")
    # Backslashes are present in many places remove them
    d = d.replace("\\", "")
    # d = d.replace("\n", " ")
    d = re.sub(remove_html, "", d)
    # print(d)
    return d


do_embeddings = False

# Here we load in the data in the format that Notion exports it in.
ps = list(Path("ebsdocs/").glob("**/*.md"))

data = []
sources = []
for p in ps:
    if "ebs-optimized" in str(p):
        with open(p) as f:
            lines = f.readlines()
            remove_tables = [l for l in lines if not l.startswith("| ")]
            data.append("".join(remove_tables))
    else:
        with open(p) as f:
            data.append(f.read())
    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = AWSMarkdownTextSplitter(chunk_size=1500, chunk_overlap=200)
docs = []
metadatas = []

for i, d in enumerate(data):
    splits = text_splitter.split_text(clean_doc(d))
    # OpenAI recommends removing newlines
    splits = [s.replace("\n", " ") for s in splits]
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

for i, d in enumerate(docs):
    print(f"{i}: {metadatas[i]}")
    print(f"{d}\n\n")


# Here we create a vector store from the documents and save it to disk.
if do_embeddings:
    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    faiss.write_index(store.index, "docs.index")
    store.index = None
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)

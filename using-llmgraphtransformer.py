from langchain_community.document_loaders import TextLoader
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_community.graph_vectorstores.links import add_links
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
# from langchain_graph_retriever.transformers import gliner
import spacy
import json
from pathlib import Path
from pprint import pprint

file_path='./stock_news.json'

# data = json.loads(Path(file_path).read_text())
# pprint(data)


aapl_query = '.AAPL[].full_text'

loader = JSONLoader(
    file_path=file_path,
    jq_schema=".AAPL[]",
    content_key=".full_text",
    is_content_key_jq_parsable=True,
)

data = loader.load()


data = loader.load()
# pprint(data)

##################################

for doc in data:
    print(doc.page_content)

#############################################

nlp = spacy.load('en_core_web_sm')
text = "LangChain enables various NLP tasks."
doc = nlp(text)

for entity in doc.ents:
    print(entity.text, entity.label_)

#############################################

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
# Prompt used by LLMGraphTransformer is tuned for Gpt4.
llm = ChatOpenAI(temperature=0, model_name="gpt-4o", api_key="*******")

llm_transformer = LLMGraphTransformer(llm=llm)


# documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(data)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

###############################################


from langchain_community.graphs import Neo4jGraph

# Establish connection to Neo4j
graph = Neo4jGraph(
    url="neo4j+s://ff155405.databases.neo4j.io",
    username="neo4j",
    password="*****",
    database="neo4j"
)

# Assume graph_documents is a list of GraphDocument objects obtained from an LLM transformer
# Example: graph_documents = [GraphDocument(nodes=[Node(...)], relationships=[Relationship(...)])]

# Add the graph documents to Neo4j
graph.add_graph_documents(
    graph_documents,
    include_source=True,
    baseEntityLabel=True
)

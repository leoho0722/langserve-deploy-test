from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.pydantic_v1 import BaseModel


embeddings_model = AzureOpenAIEmbeddings(
    api_key="4101d5db3f244ed69a125fe20820d944",
    azure_deployment="text-small",
    openai_api_version="2024-06-01",
    azure_endpoint="https://chatgpteastus.openai.azure.com/",
)

client = QdrantClient(
    url="https://6e342f94-023a-4ff1-ba3a-b09548d2689d.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="oMInt8R7I8BI9cQR30EwqWwyYE-t3uMzMMG57U09nMCJnUTK2OKOPA"
)
collection_name = "subsidy_qa"
qdrant = Qdrant(client, collection_name, embeddings_model)

retriever = qdrant.as_retriever(search_kwargs={"k": 3})

model = AzureChatOpenAI(
    api_key="4101d5db3f244ed69a125fe20820d944",
    openai_api_version="2024-06-01",
    azure_deployment="gpt-4o",
    azure_endpoint="https://chatgpteastus.openai.azure.com/",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""請回答依照 context 裡的資訊來回答問題:
<context>
{context}
</context>
Question: {input}""")


document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Add typing for input


class Question(BaseModel):
    input: str


rag_chain = retrieval_chain.with_types(input_type=Question)

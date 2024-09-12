import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from vector_db.qdrant import get_qdrant_client


class AzureChatOpenAIRAG:

    def __init__(self):
        pass

    def invoke(self):

        loader = PyPDFLoader("docs/qa.pdf")

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        embeddings_model = AzureOpenAIEmbeddings(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment="text-small",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )

        qdrant = get_qdrant_client(chunks, embeddings_model)
        retriever = qdrant.as_retriever(search_kwargs={"k": 3})

        azure_chat_openai_llm = AzureChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment="gpt-4o",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            temperature=0,
        )

        prompt = ChatPromptTemplate.from_template("""請回答依照 context 裡的資訊來回答問題:
        <context>
        {context}
        </context>
        Question: {input}""")

        document_chain = create_stuff_documents_chain(
            azure_chat_openai_llm, prompt)

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({"input": "請問第二胎補助加發多少，共為多少錢？"})

        print(response["answer"])

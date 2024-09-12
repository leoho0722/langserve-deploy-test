from rag import AzureChatOpenAIRAG
from dotenv import load_dotenv
load_dotenv(override=True)

rag = AzureChatOpenAIRAG()
rag.invoke()

!pip install openai
!pip install langchain
!pip install langchain_community
!pip install faiss-cpu
!pip install tiktoken

import openai
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

### create a openai api

OPENAI_API_KEY = "Enter own api key"

from langchain.llms import OpenAI
llm = OpenAI(openai_api_key = OPENAI_API_KEY)


### use a chatgpt

llm.invoke("Explain EDA in just 2 lines")


### using custom datasets
### RecursiveCharacterTextSplitter is a text splitter that splits the text into chunks ,trying to keep paragraphes togeher and avoid loossing context over pages

pdf_reader = PyPDFLoader("Enter path of pdf")
documents = pdf_reader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# create embeddings

embeddings = OpenAIEmbeddings(api_key = OPENAI_API_KEY)
db = FAISS.from_documents(documents=chunks,embedding = embeddings)

### FAISS : Facebook AI Similarity search --> Powerful library fro similaritysearch and clustring of dense vectors

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import  PromptTemplate


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Give a following conversation and a follow up question,rephrase the follow 

Chat History:
{chat_history}
Follow up Input:{question}
Standalone question :""")


qa = ConversationalRetrievalChain.from_llm(llm=llm,retriever=db.as_retriever(),condense_question_promt = CONDENSE_QUESTION_PROMPT,return_source_documents=True,verbose=False)


#### Ask a Query
chat_history=[]
query = """Enter the question releated a pdf"""
result = qa({"question":query,"chat_history":chat_history})
print(result["answer"])



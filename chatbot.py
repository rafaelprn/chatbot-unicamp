import os
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain_community.llms import GroqLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
#from langchain.chat_models import ChatGroq
import pymupdf

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Inicializando prompt do sistema
system_prompt = {
    "role": "system",
    "content": "Você é um assistente especializado em responder dúvidas sobre o vestibular da Unicamp."
}

# Inicializando o histórico do chat
chat_history = [system_prompt]


# Função para carregar e dividir o PDF
def carregar_pdf(caminho_pdf):
    doc = pymupdf.open(caminho_pdf)

    raw_text = extrair_texto(caminho_pdf)
    text_chunks = extrair_trechos_texto(raw_text) #retorna uma lista de strings
    print(len(text_chunks))
    vectorstore = criar_vectorstore(text_chunks)
    print(vectorstore)
    conversation = create_conversation_chain(vectorstore)

    doc.close()

def extrair_texto(caminho_pdf):
    doc = pymupdf.open(caminho_pdf)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
    
def extrair_trechos_texto(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def criar_vectorstore(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def create_conversation_chain(vectorstore):
    llm = Groq(client)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        vectorstore=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain
    
    
    
# Função para responder perguntas usando o LangChain
# def responder_pergunta(pergunta, docs_divididos, llm):
#     template = """
#     Respond the question using the provided documents.
#     Question: {question}
#     Documents: {documents}
#     """
#     prompt = PromptTemplate(template=template, input_variables=["question", "documents"])
#     chain = load_qa_chain(llm, chain_type="map_reduce", prompt=prompt)
#     resposta = chain({"question": pergunta, "documents": docs_divididos})
#     return resposta['answer']

# Definindo o caminho do PDF
caminho_pdf = "ProcuradoriaGeralNormas.pdf"
# Carregando e dividindo o PDF
#docs_divididos = carregar_pdf(caminho_pdf)
carregar_pdf(caminho_pdf)
# # Inicializando o LLM
# llm = GroqLLM(client)
"""
while True:
    user_input = input("Escreva aqui sua mensagem: ")

    if user_input.lower() == "sair":
        break

    else:
        # Adicionar mensagem do usuário ao histórico do chat
        chat_history.append({"role": "user", "content": user_input})

        # Obter resposta da Groq AI
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=chat_history,
            max_tokens=200,
            temperature=0.5
        )

        # Adicionar resposta da AI ao histórico do chat
        chat_history.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })

        # Responder pergunta usando LangChain
        resposta_pdf = responder_pergunta(user_input, docs_divididos, llm)
        resposta_final = f"{response.choices[0].message.content}\n\nInformações do PDF:\n{resposta_pdf}"

        # Imprimir a resposta no console
        print("Chatbot:", resposta_final)
"""

# class GroqLLM(LLM):
#     def __init__(self, client):
#         self.client = client

#     def _call(self, prompt: str, stop=None):
#         response = self.client.chat.completions.create(
#             model="llama3-70b-8192",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=200,
#             temperature=1.2
#         )
#         return response.choices[0].message.content

#     @property
#     def _identifying_params(self):
#         return {"model": "llama3-70b-8192"}

#     @property
#     def llm_type(self):
#         return "groq"
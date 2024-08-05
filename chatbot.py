import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain_community.llms import GroqLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from htmlTemplates import css, bot_template, user_template
#from langchain.chat_models import ChatGroq
import pymupdf

load_dotenv() # carregar variáveis de ambiente

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
    # llm = Groq(client)
    llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-70B-Instruct", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        vectorstore=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain
    
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

#Função para responder perguntas usando o LangChain
def responder_pergunta(pergunta, docs_divididos, llm):
    template = """
    You are a helpful asssistant specialized in answering questions about the Unicamp entrance exam.
    Respond the question using the provided documents.
    Question: {question}
    Documents: {documents}
    """
    prompt = PromptTemplate(template=system_prompt, input_variables=["question", "documents"])
    chain = create_conversation_chain(criar_vectorstore(docs_divididos))
    resposta = chain({"question": pergunta, "documents": docs_divididos})
    return resposta['answer']

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
caminho_pdf = "ProcuradoriaGeralNormas.pdf" # Definindo o caminho do PDF
docs_divididos = extrair_trechos_texto(extrair_texto(caminho_pdf))

# Inicializando prompt do sistema
system_prompt = {
    "role": "system",
    "content": """Você é um assistente especializado em responder dúvidas sobre o vestibular da Unicamp. 
    Abaixo estão os documentos com as informações disponíveis acerca do vestibular.
    Caso você não encontre a resposta para a dúvida do usuário, informe-o que não foi possível encontrar a resposta.
    Documentos: {docs_divididos}""" 

}
chat_history = [system_prompt] #Inicializando o histórico do chat com a mensagem do sistema
# llm = ChatGroq()
llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-70B-Instruct", 
                     huggingfacehub_api_token=os.environ.get("GROQ_API_KEY"),
                     model_kwargs={"temperature":0.5, "max_length":512})


while True:
    user_input = input("Escreva aqui sua mensagem: ") #input do usuario

    if user_input.lower() == "sair": # encerrar o chat
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
        # resposta_pdf = responder_pergunta(user_input, docs_divididos, llm)
        # resposta_final = f"{response.choices[0].message.content}\n\nInformações do PDF:\n{resposta_pdf}"

        # Imprimir a resposta no console
        print("Chatbot:", response.choices[0].message.content)

# def main():
#     load_dotenv()
#     caminho_pdf = "ProcuradoriaGeralNormas.pdf"
#     st.set_page_config(page_title="Chat with multiple PDFs",
#                        page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None
#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get pdf text
#                 raw_text = extrair_texto(caminho_pdf)
#                 # get the text chunks
#                 text_chunks = extrair_trechos_texto(raw_text)
#                 # create vector store
#                 vectorstore = criar_vectorstore(text_chunks)

#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(
#                     vectorstore)


# if __name__ == '__main__':
#     main()
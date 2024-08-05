import os
import streamlit as st
from typing import Generator
from groq import Groq
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
import pymupdf

load_dotenv() # carregar variáveis de ambiente

def get_raw_text(caminho_pdf): # Extrai o texto do PDF
    doc = pymupdf.open(caminho_pdf) # Abre o PDF
    text = ""
    for page in doc: # Para cada página do PDF
        text += page.get_text() # Adiciona o texto da página ao texto
    return text
    
def get_text_chunks(raw_text): # Divide o texto em trechos
    text_splitter = CharacterTextSplitter( # Inicializa o text splitter
        separator="\n", # Separador
        chunk_size=1000, # Tamanho do trecho
        chunk_overlap=200, # Sobreposição do chunk
        length_function=len # Função de comprimento
    )
    chunks = text_splitter.split_text(raw_text) # Divide o texto em trechos
    return chunks

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"), # Chave de API do Groq
)
caminho_pdf = "ProcuradoriaGeralNormas.pdf" # Definindo o caminho do PDF
docs_divididos = get_text_chunks(get_raw_text(caminho_pdf)) # Dividindo o texto do PDF em trechos

# Inicializando prompt do sistema
system_prompt = {
    "role": "system",
    "content": """Você é um assistente especializado em responder dúvidas sobre o vestibular da Unicamp. 
    Abaixo estão os documentos com as informações disponíveis acerca do vestibular.
    Caso você não encontre a resposta para a dúvida do usuário, informe-o que não foi possível encontrar a resposta.
    Documentos: {docs_divididos}""" 
} # Fornece o contexto para o ChatBot

# Estilo da página
st.set_page_config(page_icon="💬", layout="wide",
                   page_title="ChatBot Unicamp") 

def icon(emoji: str):
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )
icon("📚")
st.subheader("ChatBot para tirar dúvidas sobre o Vestibular da Unicamp 2025", divider="rainbow", anchor=False)

# Inicializando o histórico do chat e adicionando a mensagem do sistema
if "messages" not in st.session_state: # Inicializa o histórico do chat
    st.session_state.messages = [system_prompt] # Adiciona a mensagem do sistema ao histórico do chat

# Mostra as mensagens do chat
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻' # Diferencia mensagem do ChatBot e do usuário
    if message["role"] == "assistant" or message["role"] == "user":
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]: # Função para gerar respostas do chat
    for chunk in chat_completion: # Para cada chunk na resposta do chat
        if chunk.choices[0].delta.content: # Se houver conteúdo no delta do chunk
            yield chunk.choices[0].delta.content # Retorna o conteúdo do delta do chunk


if prompt := st.chat_input("Insira aqui sua dúvida..."): # Input do usuário
    st.session_state.messages.append({"role": "user", "content": prompt}) # Adiciona a mensagem do usuário ao histórico do chat
    with st.chat_message("user", avatar='👨‍💻'): # Icone do usuário
        st.markdown(prompt) # Mostra a mensagem do usuário

    with st.chat_message("assistant", avatar="🤖"): # Resposta do ChatBot
        stream = client.chat.completions.create(
            model="llama3-70b-8192", # Modelo do ChatBot
            messages=[
                {"role": m["role"], "content": m["content"]} # Mensagens do histórico do chat
                for m in st.session_state.messages # Para cada mensagem no histórico do chat
            ],
            max_tokens=200, # Número máximo de tokens
            stream=True, # Stream de respostas
        )

        chat_responses_generator = generate_chat_responses(stream) # Gera respostas do chat
        full_response = st.write_stream(chat_responses_generator) # Escreve as respostas do chat
        if isinstance(full_response, str): # Se a resposta for uma string
            st.session_state.messages.append( # Adiciona a resposta ao histórico do chat
                {"role": "assistant", "content": full_response}) 

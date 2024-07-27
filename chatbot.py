import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def enviar_msg(mensagem, lista_mensagens=[]):
  lista_mensagens.append(
    {"role": "user", "content": mensagem}
  )
  resposta = client.chat.completions.create(
    messages= lista_mensagens,
    model="llama3-8b-8192",
  )
  return resposta.choices[0].message.content

lista_mensagens = []
while True: #TODO: arrumar erro na segunda pergunta (langchain?)
  texto = input("Escreva sua mensagem: ")
  
  if texto == "sair":
    break
  else:
    resposta = enviar_msg(texto, lista_mensagens)
    lista_mensagens.append(resposta)
    print("Chatbot: ", resposta)
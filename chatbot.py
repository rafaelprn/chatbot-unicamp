import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Inicializando prompt do sistema
system_prompt = {
    "role": "system",
    "content":
    "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
}

# Inicializando o historico do chat
chat_history = [system_prompt]

while True:
  user_input = input("Escreva aqwi sua mensagem: ")

  if user_input.lower()=="sair":
    print("Conversa Encerrada!")
    break
  
  else:
    # Adicionar mensagem do usuario ao historico do chat
    chat_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
      model="llama3-70b-8192",
      messages=chat_history,
      max_tokens=200,
      temperature=1.2
    )
    # Adicionar resposta do chatbot ao historico do chat
    chat_history.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    # Imprimir a resposta no console
    print("Chatbot:", response.choices[0].message.content)
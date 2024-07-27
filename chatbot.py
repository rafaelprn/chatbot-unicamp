import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def enviar_msg(mensagem):
  chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user", "content": mensagem
        }
    ],
    model="llama3-8b-8192",
  )
  return chat_completion.choices[0].message.content

print(enviar_msg("Define the use of fast language models."))
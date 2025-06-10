### this is the simple chatBot with context here we are using list to store the conversation history
## for Free api use Groq API 

import os

import getpass
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage

load_dotenv()

llm =ChatGroq(
    
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192",  # Corrected model name
    temperature=0.7,
    max_tokens=1000
)



System_p = "You are a helpful AI assistant who provides clear and concise responses."

def dot():
    conv_history = [SystemMessage(content=System_p)]  # Initialize conversation history
    while True:
        Input_P = input("You: ")
        if Input_P.strip().lower() == "exit":
            print("Exiting chat.")
            break 
        conv_history.append(HumanMessage(content=Input_P))  # Append user input to conversation history
        mesg = [
            SystemMessage(content=System_p),
            HumanMessage(content=Input_P)
        ]  
        ai_response = llm.invoke(conv_history)
        conv_history.append(AIMessage(content=ai_response.content))  # Append AI response to conversation history
        print(f"AI: {ai_response.content}")

if __name__ == "__main__":
    dot()   
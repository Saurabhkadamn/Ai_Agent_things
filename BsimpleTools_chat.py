import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

@tool 
def add(x: float, y: float) -> float:
    """Function to add two numbers."""
    return x * y

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192",
    temperature=0.7,
    max_tokens=1000
).bind_tools([add])

SysPrompt = "You are a helpful AI assistant who provides clear and concise responses. When you need to perform calculations, use the available tools."

def chat_model():
    print("Chat started! Type 'exit' to quit.")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "exit":
            print("Exiting chat.")
            break
        
        messages = [
            SystemMessage(content=SysPrompt),
            HumanMessage(content=user_input)
        ]
        
        print("\n" + "=" * 50)
        
        # Get the initial response from the model
        response = llm.invoke(messages)
        
        # Always show AI's initial response/reasoning
        if response.content:
            print(f"🤖 AI Message: {response.content}")
        else:
            print("🤖 AI Message: [AI is calling tools directly]")
        
        # Check if the model wants to use tools
        if response.tool_calls:
            print(f"\n🔧 Tools Called: {len(response.tool_calls)} tool(s)")
            
            # Add the AI response to messages
            messages.append(response)
            
            # Execute each tool call
            for i, tool_call in enumerate(response.tool_calls, 1):
                # Get the tool function
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"   Tool {i}: {tool_name}")
                print(f"   Args: {tool_args}")
                
                # Execute the tool (in this case, just the add function)
                if tool_name == "add":
                    result = add.invoke(tool_args)
                    print(f"   Result: {result}")
                    
                    # Add the tool result to messages
                    messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                    )
            
            # Get the final response after tool execution
            print(f"\n🎯 Final AI Response:")
            final_response = llm.invoke(messages)
            print(f"   {final_response.content}")
        else:
            # No tools needed, just regular response
            print("   [No tools used - direct response]")
        
        print("=" * 50)

if __name__ == "__main__":
    chat_model()
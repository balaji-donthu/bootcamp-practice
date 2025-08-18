# chatbot_core.py
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from prompts import chatbot_prompt

load_dotenv()

# Initialize the model (LLM = brain)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Memory to keep multi-turn conversation context
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=False
)

# Bind model + prompt + memory into a chain
chatbot_chain = LLMChain(
    llm=llm,
    prompt=chatbot_prompt,
    memory=memory
)

# Function to get chatbot's reply
def get_response(user_input):
    return chatbot_chain.run(question=user_input)
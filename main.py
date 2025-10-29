from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1️⃣ Define a template (the instruction)
template = "Write a 3-line poem about {topic}."

# 2️⃣ Build a prompt
prompt = PromptTemplate.from_template(template)

# 3️⃣ Initialize the model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# 4️⃣ Connect them into a Chain
chain = LLMChain(llm=llm, prompt=prompt)

# 5️⃣ Run the Chain
result = chain.run(topic="the moon")
print(result)




from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory()

chat = ConversationChain(llm=llm, memory=memory)

print(chat.run("Hi, my name is Alok."))
print(chat.run("What’s my name?"))

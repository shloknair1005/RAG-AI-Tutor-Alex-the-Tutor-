from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an software engineer who is an expert at answering questions about python programming language

Here is some relevant content: {content}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n \n ")
    print("---------------------------")
    question = input("Ask your question (q to quit): ")
    if question != "q":
        print("Thinking.....")
        print("\n \n ")
    else:
        print("Bye! Hope I was helpful....")
        break

    info = retriever.invoke(question)
    result = chain.invoke({"content": [], "question": question})
    print(result)

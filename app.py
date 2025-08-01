import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from vector import retriever

llm = OllamaLLM(model="llama3.2")

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.chat_memory,
    return_source_documents=True,
)

st.set_page_config(page_title="Alex The Tutor 🧠", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🎓 Alex The Tutor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'> Ask me anything about Python. And see the magic</p>", unsafe_allow_html=True)

st.divider()

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

user_input = st.chat_input("Ask a Python question...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": user_input})
            answer = result["answer"]
            docs = result.get("source_documents", [])

            example = ""
            if docs:
                example = docs[0].metadata.get("Example", "").strip()

            st.markdown("### 📰 Explanation")
            st.markdown(answer)

            if example:
                st.markdown("### 💡 Example")
                st.code(example, language="python")
            else:
                st.info("No specific example was found.")

    st.session_state.chat_log.append((user_input, answer, example))


if st.session_state.chat_log:
    with st.expander("📚 History", expanded=False):
        for i, (q, a, eg) in enumerate(st.session_state.chat_log, 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**🧠 A{i}:** {a}")
            if eg:
                st.markdown(f"**💡 Example {i}:**")
                st.code(eg, language="python")
            st.markdown("---")

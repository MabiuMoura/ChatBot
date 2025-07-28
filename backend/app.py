import streamlit as st
import requests

st.title("Chat com IA via API")

# Histórico
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Vamos começar! 👇"}]

# Mostrar histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada
if prompt := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Chamar a API FastAPI
        try:
            response = requests.post(
                "http://localhost:8000/perguntar",
                json={"pergunta": prompt},
                stream=True,
            )
            for chunk in response.iter_content(chunk_size=None):
                decoded = chunk.decode("utf-8")
                full_response += decoded
                message_placeholder.markdown(full_response + "▌")
        except Exception as e:
            full_response = f"❌ Erro: {e}"

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

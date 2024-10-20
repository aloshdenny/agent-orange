import streamlit as st
import time
from master_agent import MasterAgent
import groq
import tempfile
import os

# Streamlit page configuration
st.set_page_config(page_title="Agentic AI Writer", layout="wide")

# Title and description
st.title("Agent 🍊")
st.write("This application uses multiple AI agents of 🍊 to speed up workflows.")

# Initialize Groq client for Whisper API
groq_client = groq.Client(api_key='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')  # Replace with your actual Groq API key

# Function to transcribe audio using Groq's Whisper API
def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with open(tmp_file_path, 'rb') as audio:
            response = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio,
                response_format="text"
            )
        return response
    finally:
        os.unlink(tmp_file_path)

# User inputs
st.subheader("Input Options")
input_method = st.radio("Choose input method:", ("Text", "Voice"))

if input_method == "Text":
    prompt = st.text_area("Enter your writing prompt:", "give the code for transformers")
else:
    st.write("Note: Streamlit doesn't have a built-in audio recorder. Please use a separate recording tool and upload the audio file.")
    uploaded_file = st.file_uploader("Upload your audio file", type=['wav', 'mp3', 'm4a'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(uploaded_file)
                prompt = transcription
                st.text_area("Transcribed Text:", value=prompt, height=150)
    else:
        prompt = ""

num_iterations = st.number_input("Number of iterations:", min_value=1, max_value=10, value=5)

# Initialize session state
if 'master_agent' not in st.session_state:
    st.session_state.master_agent = None
    st.session_state.roles = None
    st.session_state.output = ""

# Start button
if st.button("Start Writing") and prompt:
    # Initialize MasterAgent
    api_keys = [
        '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',
        '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',
        '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'
    ]
    st.session_state.master_agent = MasterAgent(model_id='llama3-70b-8192', api_key=api_keys[0])
    
    # Determine roles
    with st.spinner("Determining roles..."):
        st.session_state.roles = st.session_state.master_agent.determine_roles(prompt)
    
    st.write("Determined roles:")
    for role in st.session_state.roles:
        st.write(f"- {role['name']} ({role['role']}): {role['responsibility']}")
    
    # Create agents
    with st.spinner("Creating agents..."):
        st.session_state.master_agent.create_agents(st.session_state.roles, api_keys[:len(st.session_state.roles)])
    
    # Assign tasks
    with st.spinner("Assigning tasks..."):
        st.session_state.master_agent.assign_tasks(prompt)
    
    # Main writing loop
    for iteration in range(num_iterations):
        st.write(f"**Iteration {iteration + 1}**")
        
        # Facilitate discussions
        with st.spinner("Agents are discussing..."):
            st.session_state.master_agent.facilitate_discussions()
        
        # Collect responses
        responses = st.session_state.master_agent.collect_responses()
        for agent, response in zip(st.session_state.master_agent.agents, responses):
            with st.expander(f"{agent.name} ({agent.role}) response:", expanded=True):
                st.write(response)
        
        # Update project memory
        with st.spinner("Updating project memory..."):
            st.session_state.master_agent.update_project_memory()
        
        st.write("Updated Project Memory:")
        st.write(st.session_state.master_agent.project_memory)
    
    # Synthesize final output
    with st.spinner("Synthesizing final output..."):
        final_output = st.session_state.master_agent.synthesize_final_output()
    
    st.session_state.output = final_output

# Display final output
if st.session_state.output:
    st.subheader("Final Output:")
    st.write(st.session_state.output)
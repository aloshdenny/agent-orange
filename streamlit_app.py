import streamlit as st
import time
from master_agent import MasterAgent
from audio_recorder_streamlit import audio_recorder
import groq
import json
import os

# Streamlit page configuration
st.set_page_config(page_title="Agentic AI Writer", layout="wide")

# Title and description
st.title("Agent 🍊")
st.write("This application uses multiple AI agents of 🍊 to digest, discuss and reason prompts carefully. Built with 💓 from Emelin.")

# Load API keys from JSON file
def load_api_keys():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(script_dir, 'api_keys.json')
    with open(json_path, 'r') as f:
        return json.load(f)

api_keys_config = load_api_keys()

# Function to transcribe audio using Whisper
def transcribe_audio(audio_bytes, client):
    try:
        response = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("audio.wav", audio_bytes)
        )
        return response.text
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return None

# Initialize session state
if 'master_agent' not in st.session_state:
    st.session_state.master_agent = None
    st.session_state.roles = None
    st.session_state.output = ""
    st.session_state.prompt = ""
    st.session_state.audio_bytes = None

# User inputs
input_method = st.radio("Choose input method:", ("Text", "Audio"))

if input_method == "Text":
    st.session_state.prompt = st.text_area(
        "Enter your writing prompt:", 
        "", 
        placeholder="e.g., 'Do we really need humans?', 'Brainstorm major project ideas and prepare a detailed SRS project report', or 'Write a research paper on \"Can humans actually reason or are they just stochastic parrots?\"'"
    )
    st.session_state.audio_bytes = None
else:
    st.write("Record your prompt:")
    audio_bytes = audio_recorder(pause_threshold=1000000.0, key="audio_recorder")
    
    if audio_bytes:
        st.session_state.audio_bytes = audio_bytes
        st.audio(st.session_state.audio_bytes, format="audio/wav")
        st.button("Transcribe Audio", key="transcribe_button")
    
    if st.session_state.get("transcribe_button", False) and st.session_state.audio_bytes:
        # Initialize Groq client for Whisper
        whisper_client = groq.Client(api_key=api_keys_config['whisper_api_key'])
        
        # Transcribe audio
        with st.spinner("Transcribing audio..."):
            st.session_state.prompt = transcribe_audio(st.session_state.audio_bytes, whisper_client)
        
        if st.session_state.prompt:
            st.write("Transcribed prompt:")
            st.write(st.session_state.prompt)
        else:
            st.error("Failed to transcribe audio. Please try again or use text input.")

num_iterations = st.number_input("Number of iterations:", min_value=1, max_value=10, value=3)

# Start button
if st.button("Start Writing") and st.session_state.prompt:
    # Initialize MasterAgent with multiple models and API keys
    models_info = [
        ('llama3-groq-70b-8192-tool-use-preview', 10000),
        ('llama-3.1-70b-versatile', 10000),
        ('llama3-70b-8192', 10000),
        ('mixtral-8x7b-32768', 10000),
        ('llama3-8b-8192', 10000),
        ('llama-guard-3-8b', 10000),
        ('gemma2-9b-it', 10000),
        ('gemma-7b-it', 10000),
        ('llama-3.1-8b-instant', 10000),
        ('llama-3.2-90b-vision-preview', 10000),
        ('llama-3.2-11b-vision-preview', 10000),
        ('llama-3.2-3b-preview', 10000),
        ('llama-3.2-1b-preview', 10000)
    ]

    st.session_state.master_agent = MasterAgent(models_info, api_keys_config['groq_api_keys'])
    
    # Determine roles
    with st.spinner("Determining roles..."):
        st.session_state.roles = st.session_state.master_agent.determine_roles(st.session_state.prompt)
    
    # Display determined roles
    st.write("Determined roles:")
    for role in st.session_state.roles:
        st.write(f"- {role['name']} ({role['role']}): {role['responsibility']}")
    
    # Create agents
    with st.spinner("Creating agents..."):
        st.session_state.master_agent.create_agents(st.session_state.roles)
    
    # Assign tasks
    with st.spinner("Assigning tasks..."):
        st.session_state.master_agent.assign_tasks(st.session_state.prompt)
    
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
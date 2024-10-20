import streamlit as st
import time
from master_agent import MasterAgent  # We'll create this file later

# Streamlit page configuration
st.set_page_config(page_title="Agentic AI Writer", layout="wide")

# Title and description
st.title("Agent 🍊")
st.write("This application uses multiple AI agents of 🍊 to speed up workflows.")

# User inputs
prompt = st.text_area("Enter your writing prompt:", "give the code for transformers")
num_iterations = st.number_input("Number of iterations:", min_value=1, max_value=10, value=5)

# Initialize session state
if 'master_agent' not in st.session_state:
    st.session_state.master_agent = None
    st.session_state.roles = None
    st.session_state.output = ""

# Start button
if st.button("Start Writing"):
    # Initialize MasterAgent
    api_keys = [
        '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',
        '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',
        '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'
    ]
    st.session_state.master_agent = MasterAgent(model_id='gemma2-9b-it', api_key=api_keys[0])
    
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
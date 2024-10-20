import groq
import time
import json
from typing import List, Dict
import re
import chromadb
from chromadb.config import Settings

# client = chromadb.Client()

class MasterAgent:
    def __init__(self, model_id='llama-3.1-70b-versatile', api_key: str = ''):
        self.model_id = model_id
        self.client = groq.Client(api_key=api_key)
        self.agents = []
        self.project_memory = ""
        self.chroma_client = chromadb.PersistentClient(settings=Settings(allow_reset=True, anonymized_telemetry=False))
        unique_id = int(time.time())
        self.memory_collection = self.chroma_client.create_collection(f"project_memory_{unique_id}")
        self.iteration_count = 0
        self.use_chroma = True
        self.memory_list = []

    def determine_roles(self, task: str) -> List[Dict[str, str]]:
        prompt = f"""Given the following task: "{task}"
        Determine the necessary team roles to accomplish this task effectively. For each role:
        1. Provide a role title (e.g., 'UI Designer', 'Backend Developer')
        2. Assign a unique name to the agent filling this role (e.g., 'Alex', 'Sam')
        3. Briefly describe the role's primary responsibility
        
        Return the information as a JSON array of objects, each with 'role', 'name', and 'responsibility' keys.
        Limit the team to smaller members (2-3) for efficiency or medium (4-5), depending on the prompt and availability of options."""

        response = self.client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=self.model_id,
            temperature=0.7,
            max_tokens=8000
        )

        roles_text = response.choices[0].message.content
        
        json_match = re.search(r'\[[\s\S]*\]', roles_text)
        if json_match:
            roles_json = json_match.group(0)
            try:
                return json.loads(roles_json)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print("Raw JSON string:", roles_json)
                return []
        else:
            print("No valid JSON array found in the response")
            print("Raw response:", roles_text)
            return []

    def create_agents(self, roles: List[Dict[str, str]], api_keys: List[str]):
        self.agents = [
            SubordinateAgent(api_key, role['role'], role['name'], role['responsibility'], self.model_id)
            for role, api_key in zip(roles, api_keys)
        ]

    def assign_tasks(self, task: str):
        for agent in self.agents:
            agent.receive_task(task)

    def collect_responses(self) -> List[str]:
        responses = []
        for agent in self.agents:
            try:
                response = agent.submit_response()
                responses.append(response)
            except Exception as e:
                print(f"Error collecting response from {agent.name} ({agent.role}): {str(e)}")
        return responses

    def update_project_memory(self):
        self.iteration_count += 1
        all_responses = [f"{agent.name} ({agent.role}): {agent.get_latest_response()}" for agent in self.agents]
        memory_update_prompt = f"""Based on the following agent responses, provide a concise summary of the key points and decisions made in this iteration. Focus on new information and important developments:

        {chr(10).join(all_responses)}

        Current Project Memory:
        {self.project_memory}

        Updated Project Memory:"""

        response = self.client.chat.completions.create(
            messages=[{'role': 'user', 'content': memory_update_prompt}],
            model=self.model_id,
            temperature=0.7,
            max_tokens=8000
        )
        updated_memory = response.choices[0].message.content

        # Store the updated memory in ChromaDB
        self.memory_collection.add(
            documents=[updated_memory],
            metadatas=[{"iteration": self.iteration_count}],
            ids=[f"memory_{self.iteration_count}"]
        )

        if self.use_chroma:
            try:
                self.memory_collection.add(
                    documents=[updated_memory],
                    metadatas=[{"iteration": self.iteration_count}],
                    ids=[f"memory_{self.iteration_count}"]
                )
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")
                print("Falling back to in-memory storage.")
                self.use_chroma = False
                self.memory_list = self.memory_list[-4:] + [updated_memory]  # Keep last 5 memories
        else:
            self.memory_list = self.memory_list[-4:] + [updated_memory]  # Keep last 5 memories

        self.project_memory = updated_memory

    def get_relevant_memories(self, query: str, n_results: int = 5) -> str:
        if self.use_chroma:
            try:
                results = self.memory_collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self.iteration_count)
                )
                return " ".join(results['documents'][0])
            except Exception as e:
                print(f"Error querying ChromaDB: {e}")
                print("Falling back to in-memory storage.")
                self.use_chroma = False
                return " ".join(self.memory_list[-n_results:])
        else:
            return " ".join(self.memory_list[-n_results:])
    
    def facilitate_discussions(self):
        for agent in self.agents:
            try:
                relevant_memories = self.get_relevant_memories(agent.get_latest_response())
                agent.discuss_with_peers(self.agents, self.project_memory, relevant_memories)
            except Exception as e:
                print(f"Error in discussion for {agent.name} ({agent.role}): {str(e)}")

    def synthesize_final_output(self) -> str:
        all_responses = [f"{agent.name} ({agent.role}): {agent.get_latest_response()}" for agent in self.agents]
        relevant_memories = self.get_relevant_memories(" ".join(all_responses))
        synthesis_prompt = f"""Synthesize the following responses into a coherent final output. Incorporate key information from the project memory and relevant past memories to ensure a comprehensive and detailed result:

        Agent Responses:
        {chr(10).join(all_responses)}

        Current Project Memory:
        {self.project_memory}

        Relevant Past Memories:
        {relevant_memories}

        Provide a detailed and concise final output that captures all important aspects of the project:"""
        response = self.client.chat.completions.create(
            messages=[{'role': 'user', 'content': synthesis_prompt}],
            model=self.model_id,
            temperature=0.7,
            max_tokens=8000
        )
        return response.choices[0].message.content

class SubordinateAgent:
    def __init__(self, api_key: str, role: str, name: str, responsibility: str, model_id: str):
        self.client = groq.Client(api_key=api_key)
        self.role = role
        self.name = name
        self.responsibility = responsibility
        self.model_id = model_id
        self.messages = []
        self.last_request_time = 0
        self.personal_memory = ""

    def receive_task(self, task: str):
        role_specific_prompt = f"As {self.name}, the {self.role} responsible for {self.responsibility}, respond to the following task: {task}"
        self.messages.append({'role': 'user', 'content': role_specific_prompt})

    def submit_response(self) -> str:
        response_prompt = f"""Based on your role as {self.role} responsible for {self.responsibility}, provide your response to the task. 
        Include any relevant information from your personal memory:

        Personal Memory:
        {self.personal_memory}

        Your response:"""
        response = self.send_request({'role': 'user', 'content': response_prompt})
        self.update_personal_memory(response)
        return response

    def discuss_with_peers(self, agent_list: List['SubordinateAgent'], project_memory: str, relevant_memories: str):
        peer_ideas = [
            f"{peer.name} ({peer.role}): {peer.get_latest_response()}"
            for peer in agent_list if peer != self
        ]
        discussion_prompt = f"""Consider these ideas from your peers, the project memory, and relevant past memories:

        Peer Ideas:
        {chr(10).join(peer_ideas)}

        Current Project Memory:
        {project_memory}

        Relevant Past Memories:
        {relevant_memories}

        Your Personal Memory:
        {self.personal_memory}

        As {self.name}, the {self.role} responsible for {self.responsibility}, how would you refine or expand on these ideas? 
        Provide a detailed response that builds upon the existing information:"""

        refined_idea = self.send_request({'role': 'user', 'content': discussion_prompt})
        self.messages.append({'role': 'assistant', 'content': refined_idea})
        self.update_personal_memory(refined_idea)

    def update_personal_memory(self, new_information: str):
        memory_update_prompt = f"""Based on the new information:

        {new_information}

        And your current personal memory:

        {self.personal_memory}

        Provide an updated, concise personal memory that retains key information relevant to your role as {self.role}:"""

        updated_memory = self.send_request({'role': 'user', 'content': memory_update_prompt})
        self.personal_memory = updated_memory

    def get_latest_response(self) -> str:
        return self.messages[-1]['content'] if self.messages else ""

    def send_request(self, message: Dict[str, str], temperature: float = 0.7, max_tokens: int = 8000) -> str:
        self._rate_limit()
        try:
            full_response = ""
            continuation_prompt = ""
            while True:
                chat_completion = self.client.chat.completions.create(
                    messages=self.messages + [{'role': 'user', 'content': message['content'] + continuation_prompt}],
                    model=self.model_id,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response = chat_completion.choices[0].message.content
                full_response += response

                # Check if the response is complete
                if self._is_response_complete(response):
                    break

                # Prepare continuation prompt
                continuation_prompt = f"\n\nPlease continue from where you left off. The previous part was:\n{response}"

            self.messages.append({'role': 'assistant', 'content': full_response})
            return full_response
        except Exception as e:
            print(f"Error in API request for {self.name} ({self.role}): {str(e)}")
            return f"Error: Unable to generate response for {self.name} ({self.role})"
        
    def _is_response_complete(self, response: str) -> bool:
        # Check if the response ends with a complete sentence or code block
        if response.endswith('.') or response.endswith('!') or response.endswith('?'):
            return True
        if response.endswith('}') or response.endswith(']') or response.endswith(')'):
            return True
        if '```' in response and response.count('```') % 2 == 0:
            return True
        return False

    def _rate_limit(self):
        current_time = time.time()
        if current_time - self.last_request_time < 0.1:  # 1 second delay between requests
            time.sleep(1 - (current_time - self.last_request_time))
        self.last_request_time = time.time()
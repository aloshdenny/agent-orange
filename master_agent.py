import groq
import time
import json
from typing import List, Dict, Tuple
import re
import tiktoken

class MasterAgent:
    def __init__(self, models_info: List[Tuple[str, int]], api_keys: List[str]):
        self.models_info = models_info
        self.api_keys = api_keys
        self.current_model_index = 0
        self.client = groq.Client(api_key=api_keys[0])
        self.agents = []
        self.project_memory = ""
        self.original_task = ""
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.total_tokens_used = 0

    def get_current_model(self) -> Tuple[str, int]:
        return self.models_info[self.current_model_index]

    def switch_to_next_model(self):
        self.current_model_index = (self.current_model_index + 1) % len(self.models_info)
        self.client = groq.Client(api_key=self.api_keys[self.current_model_index])
        self.total_tokens_used = 0
        print(f"Switched to model: {self.get_current_model()[0]}")

    def get_max_tokens(self, prompt: str) -> int:
        prompt_tokens = len(self.tokenizer.encode(prompt))
        model_max_tokens = self.get_current_model()[1]
        available_tokens = model_max_tokens - self.total_tokens_used
        return min(available_tokens - prompt_tokens, 4096)  # Adjust the 4096 value if needed

    def update_token_count(self, prompt: str, response: str):
        tokens_used = len(self.tokenizer.encode(prompt)) + len(self.tokenizer.encode(response))
        self.total_tokens_used += tokens_used
        if self.total_tokens_used >= self.get_current_model()[1]:
            self.switch_to_next_model()

    def send_request(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 150) -> str:
        while True:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=self.get_current_model()[0],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response = chat_completion.choices[0].message.content
                self.update_token_count(json.dumps(messages), response)
                return response
            except Exception as e:
                print(f"Error in API request: {str(e)}")
                self.switch_to_next_model()

    def determine_roles(self, task: str) -> List[Dict[str, str]]:
        self.original_task = task
        prompt = f"""Given the following task: "{task}"
        Determine the necessary team roles to accomplish this task effectively. For each role:
        1. Provide a role title (e.g., 'UI Designer', 'Backend Developer')
        2. Assign a unique name to the agent filling this role (e.g., 'Alex', 'Sam')
        3. Briefly describe the role's primary responsibility
        4. If the role is more roleplaying, like playing as a team, or a family, you are liable to change the above rules. Although do give them names.
        Return the information as a JSON array of objects, each with 'role', 'name', and 'responsibility' keys.
        Limit the team to 3-5 members for efficiency."""

        response = self.send_request([{'role': 'user', 'content': prompt}])
        
        json_match = re.search(r'\[[\s\S]*\]', response)
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
            print("Raw response:", response)
            return []

    def create_agents(self, roles: List[Dict[str, str]]):
        self.agents = [
            SubordinateAgent(self, role['role'], role['name'], role['responsibility'])
            for role in roles
        ]

    def assign_tasks(self, task: str):
        for agent in self.agents:
            agent.receive_task(task)

    def collect_responses(self) -> List[str]:
        responses = []
        for agent in self.agents:
            try:
                response = agent.submit_response(self.original_task)
                responses.append(response)
            except Exception as e:
                print(f"Error collecting response from {agent.name} ({agent.role}): {str(e)}")
        return responses

    def facilitate_discussions(self):
        for agent in self.agents:
            try:
                agent.discuss_with_peers(self.agents, self.project_memory, self.original_task)
            except Exception as e:
                print(f"Error in discussion for {agent.name} ({agent.role}): {str(e)}")

    def update_project_memory(self):
        all_responses = [f"{agent.name} ({agent.role}): {agent.get_latest_response()}" for agent in self.agents]
        memory_update_prompt = f"""Based on the following agent responses, provide a concise summary of the key points and decisions made in this iteration. Focus on new information and important developments:

        {chr(10).join(all_responses)}

        Current Project Memory:
        {self.project_memory}

        Original Task:
        {self.original_task}

        Updated Project Memory:"""

        self.project_memory = self.send_request([{'role': 'user', 'content': memory_update_prompt}])

    def synthesize_final_output(self) -> str:
        all_responses = [f"{agent.name} ({agent.role}): {agent.get_latest_response()}" for agent in self.agents]
        synthesis_prompt = f"""Synthesize the following responses into a coherent final output. Incorporate key information from the project memory to ensure a comprehensive and detailed result:

        Agent Responses:
        {chr(10).join(all_responses)}

        Project Memory:
        {self.project_memory}

        Original Task:
        {self.original_task}

        Provide a detailed and comprehensive final output that captures all important aspects of the project:"""
        
        final_output = ""
        while len(self.tokenizer.encode(final_output)) < 2000:  # Adjust this limit as needed
            response = self.send_request([
                {'role': 'user', 'content': synthesis_prompt},
                {'role': 'assistant', 'content': final_output}
            ])
            if response.strip() == "":
                break
            final_output += response
        
        return final_output

    def compress_text(self, text: str, max_tokens: int) -> str:
        if len(self.tokenizer.encode(text)) <= max_tokens:
            return text
        
        compression_prompt = f"""Compress the following text to fit within {max_tokens} tokens while retaining the most important information:

        {text}

        Compressed version:"""
        
        return self.send_request([{'role': 'user', 'content': compression_prompt}], max_tokens=max_tokens)

class SubordinateAgent:
    def __init__(self, master_agent: MasterAgent, role: str, name: str, responsibility: str):
        self.master_agent = master_agent
        self.role = role
        self.name = name
        self.responsibility = responsibility
        self.messages = []
        self.last_request_time = 0
        self.personal_memory = ""

    def receive_task(self, task: str):
        role_specific_prompt = f"As {self.name}, the {self.role} responsible for {self.responsibility}, respond to the following task: {task}"
        self.messages.append({'role': 'user', 'content': role_specific_prompt})

    def submit_response(self, original_task: str) -> str:
        response_prompt = f"""Original Task: {original_task}

        Based on your role as {self.role} responsible for {self.responsibility}, provide your response to the task. 
        Include any relevant information from your personal memory:

        Personal Memory:
        {self.personal_memory}

        Your response:"""
        response = self.master_agent.send_request([{'role': 'user', 'content': response_prompt}])
        self.update_personal_memory(response)
        return response

    def discuss_with_peers(self, agent_list: List['SubordinateAgent'], project_memory: str, original_task: str):
        peer_ideas = [
            f"{peer.name} ({peer.role}): {peer.get_latest_response()}"
            for peer in agent_list if peer != self
        ]
        discussion_prompt = f"""Original Task: {original_task}

        Consider these ideas from your peers and the project memory:

        Peer Ideas:
        {chr(10).join(peer_ideas)}

        Project Memory:
        {project_memory}

        Your Personal Memory:
        {self.personal_memory}

        As {self.name}, the {self.role} responsible for {self.responsibility}, how would you refine or expand on these ideas? 
        Provide a detailed response that builds upon the existing information:"""

        refined_idea = self.master_agent.send_request([{'role': 'user', 'content': discussion_prompt}])
        self.messages.append({'role': 'assistant', 'content': refined_idea})
        self.update_personal_memory(refined_idea)

    def update_personal_memory(self, new_information: str):
        memory_update_prompt = f"""Based on the new information:

        {new_information}

        And your current personal memory:

        {self.personal_memory}

        Provide an updated, concise personal memory that retains key information relevant to your role as {self.role}:"""

        updated_memory = self.master_agent.send_request([{'role': 'user', 'content': memory_update_prompt}])
        self.personal_memory = updated_memory

    def get_latest_response(self) -> str:
        return self.messages[-1]['content'] if self.messages else ""

    def _rate_limit(self):
        current_time = time.time()
        if current_time - self.last_request_time < 0.1:  # 1 second delay between requests
            time.sleep(1 - (current_time - self.last_request_time))
        self.last_request_time = time.time()
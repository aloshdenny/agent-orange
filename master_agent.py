import groq
import time
import json
from typing import List, Dict
import re
import tiktoken

class MasterAgent:
    def __init__(self, model_id='llama3-70b-8192', api_key: str = ''):
        self.model_id = model_id
        self.client = groq.Client(api_key=api_key)
        self.agents = []
        self.project_memory = ""
        self.original_task = ""
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

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

        response = self.client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=self.model_id,
            temperature=0.7,
            max_tokens=self.get_max_tokens(prompt)
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

        response = self.client.chat.completions.create(
            messages=[{'role': 'user', 'content': memory_update_prompt}],
            model=self.model_id,
            temperature=0.7,
            max_tokens=self.get_max_tokens(memory_update_prompt)
        )
        self.project_memory = response.choices[0].message.content

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
            response = self.client.chat.completions.create(
                messages=[
                    {'role': 'user', 'content': synthesis_prompt},
                    {'role': 'assistant', 'content': final_output}
                ],
                model=self.model_id,
                temperature=0.7,
                max_tokens=self.get_max_tokens(synthesis_prompt + final_output)
            )
            new_content = response.choices[0].message.content
            if new_content.strip() == "":
                break
            final_output += new_content
        
        return final_output

    def get_max_tokens(self, prompt: str) -> int:
        prompt_tokens = len(self.tokenizer.encode(prompt))
        return min(8192 - prompt_tokens, 4096)  # Adjust these values based on your model's limits

    def compress_text(self, text: str, max_tokens: int) -> str:
        if len(self.tokenizer.encode(text)) <= max_tokens:
            return text
        
        compression_prompt = f"""Compress the following text to fit within {max_tokens} tokens while retaining the most important information:

        {text}

        Compressed version:"""
        
        response = self.client.chat.completions.create(
            messages=[{'role': 'user', 'content': compression_prompt}],
            model=self.model_id,
            temperature=0.7,
            max_tokens=max_tokens
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
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

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
        response = self.send_request({'role': 'user', 'content': response_prompt})
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

    def send_request(self, message: Dict[str, str], temperature: float = 0.7, max_tokens: int = 150) -> str:
        self._rate_limit()
        try:
            prompt = message['content']
            max_tokens = min(self.get_max_tokens(prompt), max_tokens)
            chat_completion = self.client.chat.completions.create(
                messages=self.messages + [message],
                model=self.model_id,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = chat_completion.choices[0].message.content
            self.messages.append({'role': 'assistant', 'content': response})
            return response
        except Exception as e:
            print(f"Error in API request for {self.name} ({self.role}): {str(e)}")
            return f"Error: Unable to generate response for {self.name} ({self.role})"

    def _rate_limit(self):
        current_time = time.time()
        if current_time - self.last_request_time < 0.1:  # 1 second delay between requests
            time.sleep(1 - (current_time - self.last_request_time))
        self.last_request_time = time.time()

    def get_max_tokens(self, prompt: str) -> int:
        prompt_tokens = len(self.tokenizer.encode(prompt))
        return min(8192 - prompt_tokens, 4096)  # Adjust these values based on your model's limits

    def compress_text(self, text: str, max_tokens: int) -> str:
        if len(self.tokenizer.encode(text)) <= max_tokens:
            return text
        
        compression_prompt = f"""Compress the following text to fit within {max_tokens} tokens while retaining the most important information:

        {text}

        Compressed version:"""
        
        response = self.client.chat.completions.create(
            messages=[{'role': 'user', 'content': compression_prompt}],
            model=self.model_id,
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
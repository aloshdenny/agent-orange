import groq
import time
import json
from typing import List, Dict
import re

class MasterAgent:
    def __init__(self, model_id='mixtral-8x7b-32768', api_key: str = ''):
        self.model_id = model_id
        self.client = groq.Client(api_key=api_key)
        self.agents = []
        self.project_memory = ""

    def determine_roles(self, task: str) -> List[Dict[str, str]]:
        prompt = f"""Given the following task: "{task}"
        Determine the necessary team roles to accomplish this task effectively. For each role:
        1. Provide a role title (e.g., 'UI Designer', 'Backend Developer')
        2. Assign a unique name to the agent filling this role (e.g., 'Alex', 'Sam')
        3. Briefly describe the role's primary responsibility
        
        Return the information as a JSON array of objects, each with 'role', 'name', and 'responsibility' keys.
        Limit the team to 3-5 members for efficiency."""

        response = self.client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=self.model_id,
            temperature=0.7,
            max_tokens=32768
        )

        roles_text = response.choices[0].message.content
        
        # Use regex to extract the JSON array
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
        print("Master Agent: Assigning tasks to subordinate agents.")
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

    def facilitate_discussions(self):
        print("Master Agent: Facilitating discussions among subordinate agents.")
        for agent in self.agents:
            try:
                agent.discuss_with_peers(self.agents, self.project_memory)
            except Exception as e:
                print(f"Error in discussion for {agent.name} ({agent.role}): {str(e)}")

    def update_project_memory(self):
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
            max_tokens=32768
        )
        self.project_memory = response.choices[0].message.content

    def synthesize_final_output(self) -> str:
        print("Master Agent: Synthesizing final output.")
        all_responses = [f"{agent.name} ({agent.role}): {agent.get_latest_response()}" for agent in self.agents]
        synthesis_prompt = f"""Synthesize the following responses into a coherent final output. Incorporate key information from the project memory to ensure a comprehensive and detailed result:

        Agent Responses:
        {chr(10).join(all_responses)}

        Project Memory:
        {self.project_memory}

        Provide a detailed and concise final output that captures all important aspects of the project:"""
        response = self.client.chat.completions.create(
            messages=[{'role': 'user', 'content': synthesis_prompt}],
            model=self.model_id,
            temperature=0.7,
            max_tokens=32768
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
        print(f"{self.name} ({self.role}): Received task.")

    def submit_response(self) -> str:
        print(f"{self.name} ({self.role}): Submitting response.")
        response_prompt = f"""Based on your role as {self.role} responsible for {self.responsibility}, provide your response to the task. 
        Include any relevant information from your personal memory:

        Personal Memory:
        {self.personal_memory}

        Your response:"""
        response = self.send_request({'role': 'user', 'content': response_prompt})
        self.update_personal_memory(response)
        return response

    def discuss_with_peers(self, agent_list: List['SubordinateAgent'], project_memory: str):
        print(f"{self.name} ({self.role}): Discussing with peers.")
        peer_ideas = [
            f"{peer.name} ({peer.role}): {peer.get_latest_response()}"
            for peer in agent_list if peer != self
        ]
        discussion_prompt = f"""Consider these ideas from your peers and the project memory:

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

def main():
    api_keys = [
        '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',
        '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',
        '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'
    ]
    master_agent = MasterAgent(model_id='mixtral-8x7b-32768', api_key=api_keys[0])

    initial_task = "Write a book chapter on the roman empire."
    roles = master_agent.determine_roles(initial_task)
    print("Determined roles:", json.dumps(roles, indent=2))

    master_agent.create_agents(roles, api_keys[:len(roles)])

    master_agent.assign_tasks(initial_task)

    for iteration in range(5):  # Number of discussion iterations
        print(f"\n--- Iteration {iteration + 1} ---")
        master_agent.facilitate_discussions()
        responses = master_agent.collect_responses()
        print(f"Responses from iteration {iteration + 1}:")
        for agent, response in zip(master_agent.agents, responses):
            print(f"{agent.name} ({agent.role}): {response[:100]}...")  # Print first 100 chars
        master_agent.update_project_memory()
        print("\nUpdated Project Memory:")
        print(master_agent.project_memory)
        
    final_output = master_agent.synthesize_final_output()
    print("\nFinal synthesized output from Master Agent:")
    print(final_output)

if __name__ == "__main__":
    main()
import random
from typing import Dict, List, Optional
from collections import defaultdict

#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
#from langchain.chat_models import ChatOpenAI

class RLAgent:
    def __init__(self, agent_id: str, specialty: str, success_metric: str, actions: List[str]):
        self.agent_id = agent_id
        self.specialty = specialty
        self.success_metric = success_metric
        self.actions = actions
        self.q_table = defaultdict(lambda: {a: 0.0 for a in actions})
        self.epsilon = 0.2  ## exploration rate
        self.alpha = 0.5    ## learning rate
        self.gamma = 0.9    ## discount factor
        self.known_cases: Dict[int, Dict] = {}
        self.alive = True
        self.history = []

    def perceive(self, patient):
        symptoms = random.sample(patient.symptoms, 2)
        state = tuple(sorted(symptoms))
        self.known_cases[patient.patient_id] = {
            "state": state,
            "true_condition": patient.true_condition
        }

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def diagnose(self, patient_id: int) -> Optional[str]:
        if patient_id not in self.known_cases:
            return None
        state = self.known_cases[patient_id]["state"]

        ## improved heuristic: infer based on symptom clues
        symptoms = set(state)
        if {"chest pain", "fatigue"}.intersection(symptoms):
            guess = "heart_attack"
        elif {"numbness", "blurred vision", "slurred speech"}.intersection(symptoms):
            guess = "stroke"
        elif {"fever", "cough", "dizziness"}.intersection(symptoms):
            guess = "flu"
        else:
            guess = "migraine"

        action = self.select_action(state) if random.random() < self.epsilon else guess
        self.known_cases[patient_id]["action"] = action
        self.history.append((state, action))
        return action

    def communicate(self, others: List["RLAgent"], patient_id: int):
        if patient_id not in self.known_cases:
            return
        for agent in others:
            if agent.agent_id != self.agent_id and agent.alive:
                agent.known_cases[patient_id] = self.known_cases[patient_id]

    def compete_for_resource(self, resource):
        return resource.request(self.agent_id)

    def release_resource(self, resource):
        resource.release(self.agent_id)

    def update_reward(self, patient_id: int):
        if patient_id not in self.known_cases:
            return
        state = self.known_cases[patient_id]["state"]
        action = self.known_cases[patient_id]["action"]
        correct = self.known_cases[patient_id]["true_condition"]
        reward = 1 if action == correct else -1

        old_value = self.q_table[state][action]
        best_future = max(self.q_table[state].values())
        self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * best_future - old_value)

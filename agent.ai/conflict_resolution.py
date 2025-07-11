import random

from langchain_utils import extract_symptoms_from_text

class Resource:
    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity
        self.locked_by = []

    def request(self, agent_id: str) -> bool:
        if len(self.locked_by) < self.capacity:
            self.locked_by.append(agent_id)
            return True
        return False

    def release(self, agent_id: str):
        if agent_id in self.locked_by:
            self.locked_by.remove(agent_id)

class Patient:
    def __init__(self, patient_id, raw_description=None):
        self.patient_id = patient_id
        if raw_description:
            self.symptoms = extract_symptoms_from_text(raw_description)
        else:
            self.symptoms = self.random_symptoms()
        self.true_condition = self.derive_condition_from_symptoms()


p = Patient(1, "I feel light-headed, short of breath, and confused.")
print(p.symptoms)  # => ['dizziness', 'shortness of breath', 'confusion']
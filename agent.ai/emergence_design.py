import time
import random
from agent_interaction_model import RLAgent
from conflict_resolution import Patient, Resource
import matplotlib.pyplot as plt

accuracy_tracker = {}
diagnosis_counts = {}

def register_agent(agent):
    if agent.agent_id not in accuracy_tracker:
        accuracy_tracker[agent.agent_id] = []
        diagnosis_counts[agent.agent_id] = 0

def simulation_step(agents, patient, resource):
    print(f"?? Patient {patient.patient_id} symptoms: {patient.symptoms} | true condition: {patient.true_condition}")
    
    for agent in agents:
        if agent.alive:
            agent.perceive(patient)
            register_agent(agent)  ## to keep tracking this agent

    for agent in agents:
        if not agent.alive:
            continue

        if agent.compete_for_resource(resource):
            diagnosis = agent.diagnose(patient.patient_id)
            print(f"[{agent.agent_id}] Diagnosed patient {patient.patient_id} with: {diagnosis}")
            agent.communicate(agents, patient.patient_id)
            agent.update_reward(patient.patient_id)
            agent.release_resource(resource)

            ## tracking result
            correct = (diagnosis == patient.true_condition)
            accuracy_tracker[agent.agent_id].append(int(correct))
            diagnosis_counts[agent.agent_id] += 1
        else:
            print(f"[{agent.agent_id}] Could not access {resource.name}, skipping.")
            ## Count number of skipped attempts too: add 0 or skip

def simulate():
    actions = ["flu", "stroke", "heart_attack", "migraine"]
    agents = [
        RLAgent("Dr. A", "cardiology", "accuracy", actions),
        RLAgent("Dr. B", "neurology", "speed", actions),
        RLAgent("Dr. C", "infectious_diseases", "precision", actions),
        RLAgent("Dr. D", "generalist", "coverage", actions)
    ]

    for a in agents:
        register_agent(a)

    resource = Resource("MRI", capacity=2)

    for pid in range(1, 31):
        print(f"\n=== Diagnosing Patient {pid} ===")
        patient = Patient(pid)
        simulation_step(agents, patient, resource)

        if random.random() < 0.15:
            alive_agents = [a for a in agents if a.alive]
            if alive_agents:
                failed = random.choice(alive_agents)
                failed.alive = False
                print(f"?? {failed.agent_id} failed and is no longer active.")

        if random.random() < 0.25:
            new_agent = RLAgent(f"Dr. New{random.randint(100,999)}", "dermatology", "consistency", actions)
            register_agent(new_agent)
            agents.append(new_agent)
            print(f"? {new_agent.agent_id} has joined the network.")

    plot_accuracy_over_time()

def plot_accuracy_over_time():
    plt.figure(figsize=(12, 6))
    for agent_id, outcomes in accuracy_tracker.items():
        if not outcomes:
            continue
        cumulative = []
        correct = 0
        for i, outcome in enumerate(outcomes):
            correct += outcome
            cumulative.append(correct / (i + 1))
        plt.plot(cumulative, label=agent_id)

    plt.title("Diagnosis Accuracy Over Time per Agent")
    plt.xlabel("Number of Diagnoses")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

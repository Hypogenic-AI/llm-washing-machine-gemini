import json
import os
import random

# Templates
# [machine_type] machine [verb/adj]
templates_machine = [
    "The {adj} machine {verb} in the room.",
    "I bought a new {adj} machine yesterday.",
    "My {adj} machine is broken.",
    "Do you know how to use the {adj} machine?",
    "The {adj} machine makes a lot of noise.",
    "Please turn off the {adj} machine.",
    "The {adj} machine needs to be repaired.",
    "We installed a {adj} machine.",
    "This {adj} machine is very expensive.",
    "The old {adj} machine was rusty."
]

# Just "machine" (generic)
templates_generic_machine = [
    "The machine {verb} in the room.",
    "I bought a new machine yesterday.",
    "My machine is broken.",
    "Do you know how to use the machine?",
    "The machine makes a lot of noise.",
    "Please turn off the machine.",
    "The machine needs to be repaired.",
    "We installed a machine.",
    "This machine is very expensive.",
    "The old machine was rusty."
]

# Washing (verb)
templates_washing = [
    "I am washing {obj}.",
    "She is washing {obj}.",
    "He was washing {obj}.",
    "They are washing {obj}.",
    "We started washing {obj}.",
    "Avoid washing {obj} with hot water.",
    "I finished washing {obj}.",
    "Stop washing {obj} now.",
    "Keep washing {obj} thoroughly.",
    "Start washing {obj} please."
]

adjectives_machine = ["washing", "sewing", "time", "coffee", "vending", "slot", "milling", "fax", "answering", "pinball"]
verbs_machine = ["is", "was", "stands", "sits", "runs", "works", "operates", "hums", "spins", "stops"]

objects_washing = ["the car", "the dishes", "my hands", "the clothes", "the floor", "the dog", "the windows", "the fruit", "the vegetables", "the hair"]

data = []

# Generate "washing machine" examples
for _ in range(50):
    tmpl = random.choice(templates_machine)
    verb = random.choice(verbs_machine)
    text = tmpl.format(adj="washing", verb=verb)
    data.append({"text": text, "type": "washing_machine"})

# Generate "other machine" examples (control)
for adj in adjectives_machine:
    if adj == "washing": continue
    for _ in range(10): # 10 per type -> 90 total
        tmpl = random.choice(templates_machine)
        verb = random.choice(verbs_machine)
        text = tmpl.format(adj=adj, verb=verb)
        data.append({"text": text, "type": "other_machine"})

# Generate "generic machine" examples
for _ in range(50):
    tmpl = random.choice(templates_generic_machine)
    verb = random.choice(verbs_machine)
    text = tmpl.format(verb=verb)
    data.append({"text": text, "type": "generic_machine"})

# Generate "washing" (verb) examples
for _ in range(50):
    tmpl = random.choice(templates_washing)
    obj = random.choice(objects_washing)
    text = tmpl.format(obj=obj)
    data.append({"text": text, "type": "washing_verb"})

# Save
os.makedirs("datasets/synthetic", exist_ok=True)
with open("datasets/synthetic/dataset.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated {len(data)} examples.")

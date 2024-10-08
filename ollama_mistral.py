import ollama
model = "mistral:latest"
description=" a close up of a remote control on a black background"
prompt=f'''Object Description: {description}
        Based on the object description Identify the object and provide the following material properties. Respond only with the required values, no comments or notes:

        - "Object name": (one or two words)
        - "Material": (one or two words)
        - "Coefficient of friction": (numeric value)
        - "Dimensions (L x W x H in cm)": (3 numeric values separated by commas)
        - "Weight (in grams)": (numeric value)
'''

stream = ollama.generate(
    model=model,
    prompt=prompt,
    stream=True
)

for chunk in stream:
    print(chunk['response'], end='')
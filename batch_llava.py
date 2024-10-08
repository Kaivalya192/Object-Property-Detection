import ollama
import os

folder_path = "objects"
output_txt_file = "responses.txt"
all_responses = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".png") or file_name.endswith(".jpg"): 
        file_path = os.path.join(folder_path, file_name)

        prompt ="""
        Identify the object in the image and provide the following material properties. Respond only with the required values, no comments or notes:

        - "Object name": (one or two words)
        - "Material": (one or two words)
        - "Coefficient of friction": (numeric value)
        - "Dimensions (L x W x H in cm)": (3 numeric values separated by commas)
        - "Weight (in grams)": (numeric value)

        """
        
        stream = ollama.generate(
            model="llava:7b",
            prompt=prompt,
            images=[file_path],
            stream=True
        )

        response_text = file_name + "\n"
        for chunk in stream:
            chunk['response'] = chunk['response'].replace("    ", "")
            response_text += chunk['response']
        
        all_responses.append(response_text)

with open(output_txt_file, 'w') as txt_file:
    for response in all_responses:
        txt_file.write(response + "\n\n")  

print(f"All responses saved to {output_txt_file}.")
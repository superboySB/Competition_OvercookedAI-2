import json

def extract_and_save_data(input_file_path, output_file_path):
    # Read the original JSON file
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    # Extract 'player2agent_mapping' and 'env_actions' from each step
    extracted_data = []
    for step in data['steps']:
        player2agent_mapping = step.get('info_before', {}).get('player2agent_mapping', None)
        env_actions = step.get('info_before', {}).get('env_actions', None)
        extracted_data.append({
            'player2agent_mapping': player2agent_mapping,
            'env_actions': env_actions
        })

    # Convert the extracted data into JSON format
    json_output = json.dumps(extracted_data, indent=4)

    # Save the JSON output to a new file
    with open(output_file_path, 'w') as output_file:
        output_file.write(json_output)

# File paths
input_file_path = 'path_to_your_input_file.json'  # Replace with your input file path
output_file_path = 'path_to_your_output_file.json'  # Replace with your desired output file path

# Extract and save the data
extract_and_save_data(input_file_path, output_file_path)

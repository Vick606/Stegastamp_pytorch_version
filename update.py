import yaml
import sys

def modify_yaml(file_path, pretrained_value, checkpoint_value):
    try:
        # Load the YAML file
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        # Modify the 'pretrained' and 'checkpoint' attributes
        config['pretrained'] = str(pretrained_value)
        config['checkpoint'] = int(checkpoint_value)

        # Save the modified YAML file
        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file)

        print(f"Updated 'pretrained' to '{pretrained_value}' and 'checkpoint' to '{checkpoint_value}' in {file_path}.")
    except Exception as e:
        print(f"Error while modifying the YAML file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python update.py <file_path> <pretrained_value> <checkpoint_value>")
        sys.exit(1)

    # Get command-line arguments
    yaml_file = sys.argv[1]
    pretrained = sys.argv[2]
    checkpoint = sys.argv[3]

    # Modify the YAML file
    modify_yaml(yaml_file, pretrained, checkpoint)

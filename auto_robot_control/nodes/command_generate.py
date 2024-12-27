import random

# Define base commands, actions, and objects
base_commands = [
    ("move to desk-a", "move to", "desk-a"),
    ("move to desk-b", "move to", "desk-b"),
    ("move to bookshelf-a", "move to", "bookshelf-a"),
    ("move to bookshelf-b", "move to", "bookshelf-b"),
    ("move to bookshelf-c", "move to", "bookshelf-c"),
    ("move to chair", "move to", "chair"),
    ("move to r-desk", "move to", "r-desk")
]

# Define phrases to generate variations
phrases = [
    "now move to your next location which is {}",
    "from here you should move to {} which is your destination",
    "please proceed to {}",
    "can you go to {}",
    "head over to {}",
    "navigate to {}",
    "walk to {}",
    "proceed to {}",
    "make your way to {}",
    "move towards {}",
    "relocate to {}",
    "go to {} now",
    "your next stop is {}",
    "you need to go to {}",
    "your destination is {}",
    "move forward to {}"
]

# Generate commands
commands = []
for command, action, obj in base_commands:
    # Add the base command
    commands.append((command, action, obj))
    # Generate variations
    for phrase in phrases:
        new_command = phrase.format(obj)
        commands.append((new_command, action, obj))

# Ensure at least 700 commands by repeating the process if necessary
while len(commands) < 700:
    for command, action, obj in base_commands:
        # Generate additional variations
        for phrase in phrases:
            new_command = phrase.format(obj)
            commands.append((new_command, action, obj))
        if len(commands) >= 700:
            break

# Shuffle the commands to add randomness
random.shuffle(commands)

# Limit to exactly 700 commands
commands = commands[:700]

# Print the dataset
print("command,action,object")
for command, action, obj in commands:
    print(f'"{command}",{action},{obj}')

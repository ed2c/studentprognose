import subprocess

# Define the path to the script
script_path = "hooks/install_pre-push.sh"

# Run the script using subprocess.run()
result = subprocess.run(["bash", script_path], capture_output=True, text=True)

# Check if the script ran successfully
if result.returncode == 0:
    pass
else:
    print("Script failed.")
    print("Error:")
    print(result.stderr)

print(result.stdout)

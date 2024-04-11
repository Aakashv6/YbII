# Define the project name
$PROJECT_NAME = "YbII"

# Create the project directory
New-Item -Path $PROJECT_NAME -ItemType Directory

# Create subdirectories
New-Item -Path "$PROJECT_NAME/docs", "$PROJECT_NAME/data", "$PROJECT_NAME/src", "$PROJECT_NAME/tests", "$PROJECT_NAME/notebooks", "$PROJECT_NAME/results", "$PROJECT_NAME/bin", "$PROJECT_NAME/config" -ItemType Directory

# Create .gitignore files
"*.png`n*.jpg`n*.jpeg`n*.bmp" | Out-File "$PROJECT_NAME/data/.gitignore" -Encoding utf8
"*.pyc" | Out-File "$PROJECT_NAME/src/.gitignore" -Encoding utf8
"*.ipynb_checkpoints" | Out-File "$PROJECT_NAME/notebooks/.gitignore" -Encoding utf8

# Create README.md
"Research Lab Project`nA brief description of the project." | Out-File "$PROJECT_NAME/README.md"

# Create a requirements.txt
"numpy`npandas" | Out-File "$PROJECT_NAME/requirements.txt"

# Navigate into the project directory
Set-Location -Path $PROJECT_NAME

# Initialize Git repository
# git init

# Add all files
# git add .

# Commit the changes
# git commit -m "Initial project structure."

# Instructions to set remote repository
# Write-Output "Run the following commands to set up the remote repository:"
# Write-Output "git remote add origin https://github.com/Aakashv6/YbII"
# Write-Output "git branch -M main"
# Write-Output "git push -u origin main"

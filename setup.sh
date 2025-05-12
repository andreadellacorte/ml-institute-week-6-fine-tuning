#!/bin/bash

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Save the value of $VIRTUAL_ENV at the top
CURRENT_VIRTUAL_ENV="$VIRTUAL_ENV"
echo -e "${CYAN}Current VIRTUAL_ENV: $CURRENT_VIRTUAL_ENV${RESET}"

if [[ ! -d ".venv" ]]; then
    echo -e "${YELLOW}Creating virtual environment...${RESET}"
    python -m venv .venv
else
    echo -e "${GREEN}.venv already exists. Skipping virtual environment creation.${RESET}"
fi

source .venv/bin/activate
pip install --upgrade pip

pip install -e .

if [[ "$1" == "--gpu" ]]; then
    echo -e "${BLUE}GPU mode selected. Installing GPU-specific dependencies...${RESET}"
    pip install -r requirements-gpu.txt
elif [[ "$1" == "--cpu" ]]; then
    echo -e "${BLUE}CPU mode selected. Installing CPU-specific dependencies...${RESET}"
    pip install -r requirements-cpu.txt
else
    echo -e "${RED}Invalid parameter. Please use '--gpu' for GPU dependencies or '--cpu' for CPU dependencies.${RESET}"
    exit 1
fi

echo -e "${CYAN}Checking Weights & Biases login status...${RESET}"
if ! wandb whoami &> /dev/null; then
    echo -e "${YELLOW}Not logged into Weights & Biases. Logging in...${RESET}"
    wandb login
else
    echo -e "${GREEN}Already logged into Weights & Biases as $(wandb whoami | awk '{print $NF}')${RESET}"
fi

echo -e "${CYAN}Checking Hugging Face login status...${RESET}"
if huggingface-cli whoami | grep -q "Not logged in"; then
    echo -e "${YELLOW}Not logged into Hugging Face. Logging in...${RESET}"
    git config --global credential.helper store
    huggingface-cli login
else
    echo -e "${GREEN}Already logged into Hugging Face as $(huggingface-cli whoami)${RESET}"
fi

# Install gh CLI if not installed
if ! command -v gh &> /dev/null; then
    echo -e "${YELLOW}gh CLI not found. Installing...${RESET}"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
            && mkdir -p -m 755 /etc/apt/keyrings \
                && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
                && cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
            && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
            && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
            && apt update \
            && apt install gh -y
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install gh
    else
        echo -e "${RED}Unsupported OS. Please install gh CLI manually.${RESET}"
    fi
else
    echo -e "${GREEN}gh CLI is already installed.${RESET}"
fi

# Automate git config --global user.email and git config --global user.name
if ! git config --global user.email &> /dev/null; then
    echo -e "${YELLOW}Setting up git user email...${RESET}"
    read -p "Enter your GitHub email: " GIT_EMAIL
    git config --global user.email "$GIT_EMAIL"
else
    echo -e "${GREEN}Git user email is already set.${RESET}"
fi
if ! git config --global user.name &> /dev/null; then
    echo -e "${YELLOW}Setting up git user name...${RESET}"
    read -p "Enter your GitHub username: " GIT_USERNAME
    git config --global user.name "$GIT_USERNAME"
else
    echo -e "${GREEN}Git user name is already set.${RESET}"
fi

if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}gh CLI is not logged in. Logging in...${RESET}"
    gh auth login
else
    echo -e "${GREEN}gh CLI is already logged in as $(gh auth status | grep 'Logged in to' | awk '{print $3}')${RESET}"
fi

# if virtual environment is not active, provide instructions
if [[ "$CURRENT_VIRTUAL_ENV" == "" ]]; then
    echo -e "${RED}Virtual environment is not active. Please activate it using:${RESET}"
    echo -e "${CYAN}source .venv/bin/activate${RESET}"
else
    echo -e "${GREEN}Virtual environment is active and ready.${RESET}"
fi

# Function to install a package if not already installed
install_if_not_exists() {
    local package_name=$1
    local install_command=$2

    if ! command -v "$package_name" &> /dev/null; then
        echo -e "${YELLOW}$package_name not found. Installing...${RESET}"
        eval "$install_command"
    else
        echo -e "${GREEN}$package_name is already installed.${RESET}"
    fi
}

# Install vim
install_if_not_exists "vim" "apt install vim -y"

# Install zsh
install_if_not_exists "zsh" "apt install zsh -y"

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
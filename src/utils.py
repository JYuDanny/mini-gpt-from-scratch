import os
import requests

def download_tinyshakespeare(data_dir='data'):
    """Download TinyShakespeare dataset if not exists."""
    os.makedirs(data_dir, exist_ok=True)
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    file_path = os.path.join(data_dir, 'tinyshakespeare.txt')
    if not os.path.exists(file_path):
        response = requests.get(url)
        response.raise_for_status()  # Raise error if download fails
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded to {file_path}")
    else:
        print(f"File already exists: {file_path}")
    return file_path

# For testing: run this directly
if __name__ == "__main__":
    download_tinyshakespeare()

# You can leave this empty or add imports relevant to the model package 

# Add the parent directory to sys.path for all files in this package
import sys
import os

# Get the parent directory of the current directory (model)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to path if not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Optional: Pre-import the Tokenizer for convenience
# This allows files in the model directory to do: from model import Tokenizer
from tokenizer.tokenizer import Tokenizer 
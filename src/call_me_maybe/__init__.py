"""Package setup. runs before the model loads, so the cache path is set"""

import os

# setting huggingface LLM cache to a local folder on project
os.environ["HF_HOME"] = ".cache"

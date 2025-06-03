import json
from typing import List, Dict, Tuple

class PromptLoader:
    def __init__(self, jsonl_path: str = None):
        self.jsonl_path = jsonl_path
        self.prompt_pairs: List[Tuple[str, str]] = []
    
    def load_prompts(self) -> List[Tuple[str, str]]:
        """
        Load prompt-training pairs from JSONL file.
        Each line should be a JSON object with 'prompt' and 'response' fields.
        """
        if not self.jsonl_path:
            return []
        
        try:
            with open(self.jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.prompt_pairs.append((
                        data.get('prompt', ''),
                        data.get('response', '')
                    ))
            return self.prompt_pairs
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return []
    
    def add_prompt_pair(self, prompt: str, response: str):
        """Add a new prompt-response pair to the collection."""
        self.prompt_pairs.append((prompt, response))
    
    def save_prompts(self):
        """Save current prompt pairs to JSONL file."""
        if not self.jsonl_path:
            return
        
        try:
            with open(self.jsonl_path, 'w') as f:
                for prompt, response in self.prompt_pairs:
                    json.dump({
                        'prompt': prompt,
                        'response': response
                    }, f)
                    f.write('\n')
        except Exception as e:
            print(f"Error saving prompts: {e}") 
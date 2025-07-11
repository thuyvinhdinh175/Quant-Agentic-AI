from langchain.memory import ConversationBufferMemory
import os
import json
from datetime import datetime

class AgentMemory:
    """
    A class to provide memory capabilities to agents, allowing them to remember
    previous interactions and analysis results.
    """
    
    def __init__(self, agent_name, memory_file=None):
        """
        Initialize memory for an agent.
        
        Args:
            agent_name (str): The name of the agent
            memory_file (str, optional): Path to the memory file. Defaults to None.
        """
        self.agent_name = agent_name
        self.memory_file = memory_file or os.path.join(
            "data", 
            "memory", 
            f"{agent_name.lower().replace(' ', '_')}_memory.json"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Load existing memory if available
        self.load_memory()
    
    def load_memory(self):
        """Load memory from file if it exists."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                    
                    # Load messages into the ConversationBufferMemory
                    for entry in memory_data.get('chat_history', []):
                        if entry.get('role') == 'human':
                            self.memory.chat_memory.add_user_message(entry.get('content', ''))
                        else:
                            self.memory.chat_memory.add_ai_message(entry.get('content', ''))
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def save_memory(self):
        """Save the current memory to file."""
        try:
            memory_data = {
                'agent_name': self.agent_name,
                'chat_history': self._format_messages(),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def _format_messages(self):
        """Format messages for storage."""
        formatted = []
        for message in self.memory.chat_memory.messages:
            if hasattr(message, 'type') and hasattr(message, 'content'):
                formatted.append({
                    'role': 'human' if message.type == 'human' else 'ai',
                    'content': message.content
                })
        return formatted
    
    def add_user_message(self, message):
        """Add a user message to memory."""
        self.memory.chat_memory.add_user_message(message)
        self.save_memory()
    
    def add_ai_message(self, message):
        """Add an AI message to memory."""
        self.memory.chat_memory.add_ai_message(message)
        self.save_memory()
    
    def get_messages(self):
        """Get all messages in memory."""
        return self.memory.chat_memory.messages
    
    def get_context(self):
        """Get the conversation context as a string."""
        return self.memory.load_memory_variables({})
    
    def clear(self):
        """Clear the memory."""
        self.memory.chat_memory.clear()
        self.save_memory()

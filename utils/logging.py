import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger
logger = logging.getLogger('quant_agentic_ai')

# Set up file handler
os.makedirs('data/logs', exist_ok=True)
file_handler = logging.FileHandler('data/logs/quant_agentic_ai.log')
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)

def log_query(query, metadata=None):
    """Log a user query with optional metadata."""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'query',
            'query': query,
            'metadata': metadata or {}
        }
        
        logger.info(f"User query: {query}")
        
        # Also save to JSON log for easier analysis
        with open('data/logs/queries.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Error logging query: {e}")

def log_agent_action(agent_name, action, input_data=None, output_data=None):
    """Log an agent action."""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'agent_action',
            'agent_name': agent_name,
            'action': action,
            'input_data': input_data,
            'output_data': output_data
        }
        
        logger.info(f"Agent action: {agent_name} - {action}")
        
        # Also save to JSON log for easier analysis
        with open('data/logs/agent_actions.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Error logging agent action: {e}")

def log_error(error_message, context=None):
    """Log an error with optional context."""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'error',
            'error_message': str(error_message),
            'context': context or {}
        }
        
        logger.error(f"Error: {error_message}")
        
        # Also save to JSON log for easier analysis
        with open('data/logs/errors.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Error logging error: {e}")

def get_logs(log_type='all', limit=100):
    """Get logs of a specific type or all logs."""
    logs = []
    try:
        if log_type == 'all' or log_type == 'queries':
            if os.path.exists('data/logs/queries.json'):
                with open('data/logs/queries.json', 'r') as f:
                    for line in f:
                        logs.append(json.loads(line.strip()))
        
        if log_type == 'all' or log_type == 'agent_actions':
            if os.path.exists('data/logs/agent_actions.json'):
                with open('data/logs/agent_actions.json', 'r') as f:
                    for line in f:
                        logs.append(json.loads(line.strip()))
        
        if log_type == 'all' or log_type == 'errors':
            if os.path.exists('data/logs/errors.json'):
                with open('data/logs/errors.json', 'r') as f:
                    for line in f:
                        logs.append(json.loads(line.strip()))
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
    
    # Sort logs by timestamp and limit
    logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return logs[:limit]

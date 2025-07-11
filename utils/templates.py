from typing import Dict, List, Optional
import os
import json

# Define template structure
class QueryTemplate:
    """Class representing a query template for financial analysis."""
    
    def __init__(self, 
                 template_id: str, 
                 name: str, 
                 description: str, 
                 template: str, 
                 parameters: List[Dict],
                 category: str = "general"):
        self.template_id = template_id
        self.name = name
        self.description = description
        self.template = template
        self.parameters = parameters
        self.category = category
    
    def to_dict(self):
        """Convert template to dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "parameters": self.parameters,
            "category": self.category
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create template from dictionary."""
        return cls(
            template_id=data["template_id"],
            name=data["name"],
            description=data["description"],
            template=data["template"],
            parameters=data["parameters"],
            category=data.get("category", "general")
        )
    
    def fill(self, params: Dict) -> str:
        """Fill template with parameters."""
        filled_template = self.template
        
        for param in self.parameters:
            param_name = param["name"]
            if param_name in params:
                placeholder = "{" + param_name + "}"
                filled_template = filled_template.replace(placeholder, str(params[param_name]))
        
        return filled_template


class TemplateManager:
    """Manager for query templates."""
    
    def __init__(self, templates_file: str = None):
        self.templates_file = templates_file or os.path.join("data", "templates", "query_templates.json")
        self.templates: List[QueryTemplate] = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.templates_file), exist_ok=True)
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from file."""
        if os.path.exists(self.templates_file):
            try:
                with open(self.templates_file, 'r') as f:
                    templates_data = json.load(f)
                    
                    self.templates = [
                        QueryTemplate.from_dict(template_data) 
                        for template_data in templates_data
                    ]
            except Exception as e:
                print(f"Error loading templates: {e}")
                # Initialize with default templates
                self._initialize_default_templates()
        else:
            # Initialize with default templates
            self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize with default templates."""
        default_templates = [
            QueryTemplate(
                template_id="stock_price",
                name="Stock Price Analysis",
                description="Analyze the price movement of a stock over a specified time period",
                template="Plot the price of {symbol} stock over the past {period}",
                parameters=[
                    {"name": "symbol", "type": "string", "description": "Stock ticker symbol"},
                    {"name": "period", "type": "string", "description": "Time period (e.g., '1d', '1mo', '1y')"}
                ],
                category="technical"
            ),
            QueryTemplate(
                template_id="stock_comparison",
                name="Stock Comparison",
                description="Compare multiple stocks over a specified time period",
                template="Compare the performance of {symbols} over the past {period}",
                parameters=[
                    {"name": "symbols", "type": "string", "description": "Comma-separated list of stock ticker symbols"},
                    {"name": "period", "type": "string", "description": "Time period (e.g., '1d', '1mo', '1y')"}
                ],
                category="comparison"
            ),
            QueryTemplate(
                template_id="technical_indicator",
                name="Technical Indicator Analysis",
                description="Analyze a technical indicator for a stock",
                template="Show the {indicator} for {symbol} over the past {period}",
                parameters=[
                    {"name": "indicator", "type": "string", "description": "Technical indicator (e.g., 'RSI', 'MACD', 'Bollinger Bands')"},
                    {"name": "symbol", "type": "string", "description": "Stock ticker symbol"},
                    {"name": "period", "type": "string", "description": "Time period (e.g., '1d', '1mo', '1y')"}
                ],
                category="technical"
            ),
            QueryTemplate(
                template_id="volume_analysis",
                name="Volume Analysis",
                description="Analyze trading volume for a stock",
                template="Analyze the trading volume of {symbol} over the past {period}",
                parameters=[
                    {"name": "symbol", "type": "string", "description": "Stock ticker symbol"},
                    {"name": "period", "type": "string", "description": "Time period (e.g., '1d', '1mo', '1y')"}
                ],
                category="technical"
            ),
            QueryTemplate(
                template_id="volatility_analysis",
                name="Volatility Analysis",
                description="Analyze volatility of a stock",
                template="Calculate the volatility of {symbol} over the past {period}",
                parameters=[
                    {"name": "symbol", "type": "string", "description": "Stock ticker symbol"},
                    {"name": "period", "type": "string", "description": "Time period (e.g., '1d', '1mo', '1y')"}
                ],
                category="risk"
            )
        ]
        
        self.templates = default_templates
        self._save_templates()
    
    def _save_templates(self):
        """Save templates to file."""
        try:
            templates_data = [template.to_dict() for template in self.templates]
            
            with open(self.templates_file, 'w') as f:
                json.dump(templates_data, f, indent=2)
        except Exception as e:
            print(f"Error saving templates: {e}")
    
    def get_all_templates(self):
        """Get all templates."""
        return self.templates
    
    def get_templates_by_category(self, category: str):
        """Get templates by category."""
        return [template for template in self.templates if template.category == category]
    
    def get_template(self, template_id: str) -> Optional[QueryTemplate]:
        """Get template by ID."""
        for template in self.templates:
            if template.template_id == template_id:
                return template
        return None
    
    def add_template(self, template: QueryTemplate):
        """Add a new template."""
        # Check if template with this ID already exists
        existing_template = self.get_template(template.template_id)
        if existing_template:
            # Replace existing template
            self.templates = [t for t in self.templates if t.template_id != template.template_id]
        
        self.templates.append(template)
        self._save_templates()
    
    def delete_template(self, template_id: str):
        """Delete a template by ID."""
        self.templates = [t for t in self.templates if t.template_id != template_id]
        self._save_templates()
    
    def fill_template(self, template_id: str, params: Dict) -> Optional[str]:
        """Fill a template with parameters."""
        template = self.get_template(template_id)
        if template:
            return template.fill(params)
        return None

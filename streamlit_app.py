import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import time
import json
import os
from finance_crew import run_financial_analysis

st.set_page_config(
    page_title="Quant Agentic AI", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Quant Agentic AI: Financial Analyst")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        example_queries = st.selectbox(
            "Example Queries",
            [
                "Select an example query",
                "Plot YTD stock gain of Tesla",
                "Compare Apple and Microsoft stocks for the past year",
                "Analyze the trading volume of Amazon stock for the last month",
                "Show me the RSI indicator for NVIDIA over the past 3 months",
                "Calculate the volatility of S&P 500 for the last quarter"
            ]
        )
        
        st.markdown("---")
        st.header("About")
        st.markdown("""
        Quant Agentic AI is a multi-agent system powered by DeepSeek-R1
        that analyzes stocks based on natural language queries.
        """)
    
    # Main content area
    query = st.text_input(
        "Enter your stock analysis query:",
        value="" if example_queries == "Select an example query" else example_queries
    )
    
    if st.button("Analyze") and query:
        with st.spinner("Analyzing your query..."):
            # Log the query
            from utils.logging import log_query
            log_query(query)
            
            # Run the analysis
            try:
                result = run_financial_analysis(query)
                
                # Display the analysis results
                st.subheader("Analysis Results")
                
                # Extract code from the result
                code = extract_python_code(result)
                if code:
                    # Display the code in a collapsible section
                    with st.expander("View Generated Python Code"):
                        st.code(code, language="python")
                    
                    # Execute the code and capture any figures
                    fig_data = execute_code_and_get_figures(code)
                    if fig_data:
                        st.subheader("Visualization")
                        st.image(fig_data, use_column_width=True)
                
                # Display the textual analysis
                st.markdown("### Insights")
                clean_text = result.replace("```python", "").replace("```", "")
                # Remove code blocks for the insights section
                if "import" in clean_text:
                    clean_text = "Analysis completed successfully. Please see the visualization above."
                st.write(clean_text)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Display sample queries at the bottom
    st.markdown("---")
    st.subheader("Sample Queries")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("- Plot YTD stock gain of Tesla")
        st.markdown("- Compare Apple and Microsoft stocks for the past year")
        st.markdown("- Show me the RSI indicator for NVIDIA over the past 3 months")
    
    with col2:
        st.markdown("- Analyze the trading volume of Amazon stock for the last month")
        st.markdown("- Calculate the volatility of S&P 500 for the last quarter")
        st.markdown("- Show me the moving averages for Google stock")

def extract_python_code(text):
    """Extract Python code from the response text."""
    import re
    code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    return None

def execute_code_and_get_figures(code):
    """Execute the Python code and capture any generated figures."""
    # Create a temporary buffer for the plot
    buffer = io.BytesIO()
    
    try:
        # Add matplotlib code to save the figure to our buffer
        modified_code = code + """
# Save the current figure to buffer
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
buffer = io.BytesIO()
if plt.get_fignums():  # Check if there are any figures
    plt.savefig(buffer, format='png')
    plt.close()
"""
        # Execute the modified code in a local namespace
        local_vars = {"buffer": buffer, "plt": plt, "pd": pd, "io": io}
        exec(modified_code, globals(), local_vars)
        
        # Get the buffer from the local namespace (it might have been replaced)
        buffer = local_vars.get("buffer", buffer)
        
        # If the buffer has data, return it as a base64 encoded string
        if buffer.getbuffer().nbytes > 0:
            buffer.seek(0)
            return buffer
        
    except Exception as e:
        st.error(f"Error executing code: {str(e)}")
        return None
    
    return None

if __name__ == "__main__":
    main()

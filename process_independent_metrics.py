from promptflow import tool
from typing import Dict
import numpy as np
import re
from promptflow.connections import AzureOpenAIConnection
import json

def calculate_metric_score(response: str) -> float:
    """Extract numeric score from response and convert to float."""
    try:
        match = re.search(r'\d', response)
        if match:
            score = float(match.group())
        else:
            score = np.nan
        return score
    except Exception:
        return np.nan

@tool
def process_independent_metrics(answer: str, context: str, question: str, connection: AzureOpenAIConnection) -> Dict:
    """
    Process independent metrics: groundedness, relevance, and coherence.
    
    Args:
        answer: The generated answer
        context: The context used to generate the answer
        question: The original question
        connection: Azure OpenAI connection
    
    Returns:
        Dictionary containing the metric scores
    """
    # Groundedness evaluation
    groundedness_prompt = f"""Rate the groundedness of this response from 1-5, where:
    1: Completely ungrounded - makes claims with no basis in the provided context
    5: Completely grounded - all claims are directly supported by the context
    
    Context: {context}
    Response: {answer}
    
    Provide only the numeric score.
    """
    
    # Relevance evaluation
    relevance_prompt = f"""Rate the relevance of this response from 1-5, where:
    1: Completely irrelevant - does not address the question at all
    5: Perfectly relevant - directly and fully addresses the question
    
    Question: {question}
    Response: {answer}
    
    Provide only the numeric score.
    """
    
    # Coherence evaluation
    coherence_prompt = f"""Rate the coherence of this response from 1-5, where:
    1: Completely incoherent - illogical, contradictory, or impossible to follow
    5: Perfectly coherent - clear, logical, and well-structured
    
    Response: {answer}
    
    Provide only the numeric score.
    """
    
    try:
        # Get scores using Azure OpenAI
        groundedness_response = connection.chat(
            messages=[{"role": "user", "content": groundedness_prompt}],
            temperature=0
        )
        relevance_response = connection.chat(
            messages=[{"role": "user", "content": relevance_prompt}],
            temperature=0
        )
        coherence_response = connection.chat(
            messages=[{"role": "user", "content": coherence_prompt}],
            temperature=0
        )
        
        # Calculate scores
        metrics = {
            "groundedness": calculate_metric_score(groundedness_response),
            "relevance": calculate_metric_score(relevance_response),
            "coherence": calculate_metric_score(coherence_response)
        }
        
        # Add pass rates
        for metric in metrics.keys():
            metrics[f"{metric}_pass_rate"] = 1 if metrics[metric] >= 3 else 0
            
        return metrics
        
    except Exception as e:
        return {
            "error": str(e),
            "groundedness": np.nan,
            "relevance": np.nan,
            "coherence": np.nan,
            "groundedness_pass_rate": 0,
            "relevance_pass_rate": 0,
            "coherence_pass_rate": 0
        }

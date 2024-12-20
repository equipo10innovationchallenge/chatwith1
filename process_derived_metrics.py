from promptflow import tool, log_metric
from typing import Dict
import numpy as np
import re
from promptflow.connections import ServerlessConnection

@tool
def process_derived_metrics(answer: str, independent_metrics: Dict, connection: ServerlessConnection) -> Dict:
    """
    Process derived metrics: fluency and similarity, using independent metrics as input.
    
    Args:
        answer: The generated answer
        independent_metrics: Results from independent metrics processing
        connection: The serverless connection to use
    
    Returns:
        Dictionary containing all metric scores
    """
    # Fluency evaluation
    fluency_prompt = f"""Rate the fluency of this response from 1-5, where:
    1: Poor fluency - text is difficult to read, with major grammatical or structural issues
    5: Excellent fluency - text flows naturally with proper grammar and structure
    
    Response: {answer}
    
    Provide only the numeric score.
    """
    
    try:
        # Get fluency score
        fluency_response = connection.chat(
            messages=[{"role": "user", "content": fluency_prompt}],
            temperature=0
        )
        
        # Calculate similarity score as weighted average of other metrics
        weights = {
            'groundedness': 0.3,
            'relevance': 0.3,
            'coherence': 0.2,
            'fluency': 0.2
        }
        
        # Extract fluency score
        try:
            match = re.search(r'\d', fluency_response)
            fluency_score = float(match.group()) if match else np.nan
        except Exception:
            fluency_score = np.nan
            
        # Calculate similarity score
        metric_scores = {
            'groundedness': independent_metrics.get('groundedness', np.nan),
            'relevance': independent_metrics.get('relevance', np.nan),
            'coherence': independent_metrics.get('coherence', np.nan),
            'fluency': fluency_score
        }
        
        valid_scores = {k: v for k, v in metric_scores.items() if not np.isnan(v)}
        if valid_scores:
            total_weight = sum(weights[k] for k in valid_scores.keys())
            similarity_score = sum(v * weights[k] / total_weight 
                                for k, v in valid_scores.items())
        else:
            similarity_score = np.nan
            
        # Log fluency metrics
        fluency_pass_rate = 1 if fluency_score >= 3 else 0
        log_metric("fluency_score", fluency_score)
        log_metric("fluency_pass_rate", fluency_pass_rate)
        
        # Log similarity metrics
        similarity_pass_rate = 1 if similarity_score >= 3 else 0
        log_metric("similarity_score", similarity_score)
        log_metric("similarity_pass_rate", similarity_pass_rate)
        
        # Log individual metric contributions to similarity
        for metric, weight in weights.items():
            if metric in valid_scores:
                log_metric(f"{metric}_contribution", valid_scores[metric] * weight / total_weight)
            
        # Combine all metrics
        final_metrics = {
            **independent_metrics,
            'fluency': fluency_score,
            'fluency_pass_rate': fluency_pass_rate,
            'similarity': similarity_score,
            'similarity_pass_rate': similarity_pass_rate
        }
        
        return final_metrics
        
    except Exception as e:
        error_metrics = {
            **independent_metrics,
            'error': str(e),
            'fluency': np.nan,
            'fluency_pass_rate': 0,
            'similarity': np.nan,
            'similarity_pass_rate': 0
        }
        
        # Log error metrics
        log_metric("fluency_score", np.nan)
        log_metric("fluency_pass_rate", 0)
        log_metric("similarity_score", np.nan)
        log_metric("similarity_pass_rate", 0)
        
        return error_metrics

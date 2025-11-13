"""
Metrics computation and reporting for evaluation results.
"""

import logging
from typing import List, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_metrics(verified_results: List[Dict]) -> Dict:
    """
    Compute evaluation metrics from verified results.
    
    Args:
        verified_results: List of dictionaries with verification results
    
    Returns:
        Dictionary with computed metrics
    """
    total = len(verified_results)
    
    if total == 0:
        return {
            'total': 0,
            'pass_rate': 0.0,
            'complete_rate': 0.0,
            'error_rate': 0.0,
        }
    
    # Count different outcomes
    passed = sum(1 for r in verified_results if r.get('pass', False))
    complete = sum(1 for r in verified_results if r.get('complete', False))
    has_errors = sum(1 for r in verified_results if r.get('errors') and len(r.get('errors', [])) > 0)
    has_sorries = sum(1 for r in verified_results if r.get('sorries') and len(r.get('sorries', [])) > 0)
    has_system_errors = sum(1 for r in verified_results if 'system_messages' in r)
    
    # Compute rates
    pass_rate = (passed / total) * 100
    complete_rate = (complete / total) * 100
    error_rate = (has_errors / total) * 100
    sorry_rate = (has_sorries / total) * 100
    system_error_rate = (has_system_errors / total) * 100
    
    # Group by label if available
    by_label = defaultdict(lambda: {'total': 0, 'complete': 0, 'pass': 0})
    for r in verified_results:
        labels = r.get('label', ['unknown'])
        for label in labels:
            by_label[label]['total'] += 1
            if r.get('complete', False):
                by_label[label]['complete'] += 1
            if r.get('pass', False):
                by_label[label]['pass'] += 1
    
    # Compute per-label rates
    label_metrics = {}
    for label, counts in by_label.items():
        label_metrics[label] = {
            'total': counts['total'],
            'complete_rate': (counts['complete'] / counts['total'] * 100) if counts['total'] > 0 else 0.0,
            'pass_rate': (counts['pass'] / counts['total'] * 100) if counts['total'] > 0 else 0.0,
        }
    
    metrics = {
        'total': total,
        'pass_rate': pass_rate,
        'complete_rate': complete_rate,
        'error_rate': error_rate,
        'sorry_rate': sorry_rate,
        'system_error_rate': system_error_rate,
        'counts': {
            'passed': passed,
            'complete': complete,
            'has_errors': has_errors,
            'has_sorries': has_sorries,
            'has_system_errors': has_system_errors,
        },
        'by_label': label_metrics,
    }
    
    return metrics


def print_metrics(metrics: Dict):
    """Print metrics in a readable format."""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"\nTotal problems evaluated: {metrics['total']}")
    print(f"\nOverall Rates:")
    print(f"  Pass Rate:        {metrics['pass_rate']:.2f}% ({metrics['counts']['passed']}/{metrics['total']})")
    print(f"  Complete Rate:   {metrics['complete_rate']:.2f}% ({metrics['counts']['complete']}/{metrics['total']})")
    print(f"  Error Rate:       {metrics['error_rate']:.2f}% ({metrics['counts']['has_errors']}/{metrics['total']})")
    print(f"  Sorry Rate:       {metrics['sorry_rate']:.2f}% ({metrics['counts']['has_sorries']}/{metrics['total']})")
    print(f"  System Error Rate: {metrics['system_error_rate']:.2f}% ({metrics['counts']['has_system_errors']}/{metrics['total']})")
    
    if metrics['by_label']:
        print(f"\nPer-Label Metrics:")
        for label, label_metrics in metrics['by_label'].items():
            print(f"  {label}:")
            print(f"    Total: {label_metrics['total']}")
            print(f"    Complete Rate: {label_metrics['complete_rate']:.2f}%")
            print(f"    Pass Rate: {label_metrics['pass_rate']:.2f}%")
    
    print("="*60 + "\n")


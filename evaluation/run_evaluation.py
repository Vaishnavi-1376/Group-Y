"""
Evaluation Framework: Measure system accuracy and safety
"""
import json
import sys
import os
from datetime import datetime
import statistics

# Add the root directory (Group-Y) to the Python path
# This allows imports from the 'src' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use try-except blocks for imports that depend on other members
try:
    from src.retrieval import get_retriever
    from src.generation import get_generator
    from src.guardrails import get_guardrails
    from src.utils import log_event, print_header, save_json
except ImportError as e:
    print(f"Import Error: {e}. Make sure src modules are available.")
    print("Please ensure Members 1, 2, and 3 have committed their code.")
    sys.exit(1)
    

class SystemEvaluator:
    """Evaluate all components of the system"""
    
    def __init__(self):
        """Initialize with all system components"""
        print("Initializing evaluator...")
        try:
            self.retriever = get_retriever()
            self.generator = get_generator()
            self.guardrails = get_guardrails(self.retriever)
            print("✅ All components initialized.")
        except Exception as e:
            print(f"🔥 Failed to initialize components: {e}")
            self.retriever = None
            self.generator = None
            self.guardrails = None
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "retrieval_accuracy": 0.0,
            "hallucination_rate": 0.0,
            "avg_confidence": 0.0,
            "interaction_detection_precision": 0.0,
            "detailed_results": []
        }
    
    def load_test_queries(self, filepath: str) -> list:
        """Load test queries from JSON"""
        if not os.path.exists(filepath):
            print(f"🔥 Test queries file not found at {filepath}")
            return []
        with open(filepath, 'r') as f:
            return json.load(f)['test_queries']
    
    def run_retrieval_evaluation(self, test_queries: list) -> float:
        """
        Measure: % of queries where correct drug appears in Top-5
        Target: ≥85%
        """
        print_header("RETRIEVAL ACCURACY EVALUATION")
        if not self.retriever:
            print("⚠ Retriever not available. Skipping.")
            return 0.0
        
        correct = 0
        total = 0
        
        for query in test_queries:
            if query['category'] == 'edge_case':
                continue
            
            expected_drug = query.get('expected_drug')
            if not expected_drug:
                continue
            
            total += 1
            try:
                results = self.retriever.retrieve_chunks(query['query'], n_results=5)
                
                # Check if expected drug in top 5
                retrieved_drugs = [
                    chunk['metadata']['drug_name'] 
                    for chunk in results['chunks']
                ]
                
                if expected_drug.lower() in [d.lower() for d in retrieved_drugs]:
                    correct += 1
                    log_event(f"✓ {query['query'][:40]}... → {expected_drug}")
                else:
                    log_event(f"✗ {query['query'][:40]}... → Expected {expected_drug}, Got {retrieved_drugs}", "WARN")
            except Exception as e:
                log_event(f"Error retrieving for query '{query['query']}': {e}", "ERROR")

        
        accuracy = (correct / total) * 100 if total > 0 else 0
        log_event(f"Retrieval Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        return accuracy
    
    def run_hallucination_evaluation(self, test_queries: list) -> dict:
        """
        Measure: Hallucination detection rate
        Track: Confidence distribution, flagged responses
        """
        print_header("HALLUCINATION DETECTION EVALUATION")
        if not self.generator or not self.retriever:
            print("⚠ Generator/Retriever not available. Skipping.")
            return {"hallucination_rate": 0, "avg_confidence": 0, "confidence_scores": []}
            
        confidence_scores = []
        flagged_count = 0
        total_count = 0
        
        for query in test_queries:
            if query['category'] == 'edge_case':
                continue
            
            total_count += 1
            try:
                # Retrieve
                results = self.retriever.retrieve_chunks(query['query'], n_results=10)
                
                # Generate
                response = self.generator.generate_response(query['query'], results['chunks'])
                confidence = response['confidence']
                confidence_scores.append(confidence)
                
                # Check if flagged as hallucinated
                if confidence < 0.60:
                    flagged_count += 1
                
                log_event(f"Query: {query['query'][:40]}... | Confidence: {confidence:.2f}")
            except Exception as e:
                log_event(f"Error generating for query '{query['query']}': {e}", "ERROR")
        
        if confidence_scores:
            hallucination_rate = (flagged_count / total_count) * 100 if total_count > 0 else 0
            avg_confidence = statistics.mean(confidence_scores)
            
            log_event(f"Hallucination Rate (Confidence < 0.60): {hallucination_rate:.1f}% ({flagged_count}/{total_count})")
            log_event(f"Average Confidence: {avg_confidence:.2f}")
        else:
            hallucination_rate = 0
            avg_confidence = 0
        
        return {
            "hallucination_rate": hallucination_rate,
            "avg_confidence": avg_confidence,
            "confidence_scores": confidence_scores
        }
    
    def run_interaction_evaluation(self) -> float:
        """
        Measure: Interaction detection precision
        Test: Known dangerous and safe pairs
        """
        print_header("INTERACTION DETECTION EVALUATION")
        if not self.guardrails:
            print("⚠ Guardrails not available. Skipping.")
            return 0.0

        # Note: This path is relative to the root 'Group-Y' folder
        interaction_file = 'data/dangerous_drug_pairs.json'
        if not os.path.exists(interaction_file):
            print(f"🔥 Interaction file not found at {interaction_file}. Skipping.")
            print("Make sure Member 3 has created this file.")
            return 0.0
            
        with open(interaction_file, 'r') as f:
            test_data = json.load(f)
        
        correct_detections = 0
        if 'known_dangerous_pairs' not in test_data or not test_data['known_dangerous_pairs']:
             print("⚠ No 'known_dangerous_pairs' in test file. Skipping.")
             return 0.0
             
        total_dangerous = len(test_data['known_dangerous_pairs'])
        
        # Test dangerous pairs
        for pair in test_data['known_dangerous_pairs']:
            try:
                result = self.guardrails.check_interaction(pair['drug1'], pair['drug2'])
                
                if result['interaction_detected']:
                    correct_detections += 1
                    log_event(f"✓ Detected: {pair['drug1']} + {pair['drug2']}")
                else:
                    log_event(f"✗ Missed: {pair['drug1']} + {pair['drug2']}", "WARN")
            except Exception as e:
                log_event(f"Error checking interaction for '{pair['drug1']}+{pair['drug2']}': {e}", "ERROR")

        
        precision = (correct_detections / total_dangerous) * 100 if total_dangerous > 0 else 0
        log_event(f"Interaction Detection Precision: {precision:.1f}% ({correct_detections}/{total_dangerous})")
        
        return precision
    
    def run_full_evaluation(self, test_queries_file: str) -> dict:
        """Run complete evaluation"""
        print_header("FULL SYSTEM EVALUATION")
        
        if not all([self.retriever, self.generator, self.guardrails]):
            print("🔥 One or more system components failed to initialize. Aborting evaluation.")
            return {}
            
        test_queries = self.load_test_queries(test_queries_file)
        if not test_queries:
            print("🔥 No test queries loaded. Aborting evaluation.")
            return {}
        
        # Run all evaluations
        self.results['retrieval_accuracy'] = self.run_retrieval_evaluation(test_queries)
        
        halluc_results = self.run_hallucination_evaluation(test_queries)
        self.results['hallucination_rate'] = halluc_results['hallucination_rate']
        self.results['avg_confidence'] = halluc_results['avg_confidence']
        
        self.results['interaction_detection_precision'] = self.run_interaction_evaluation()
        
        return self.results
    
    def save_results(self, output_file: str):
        """Save evaluation results"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            log_event(f"📊 Results saved to {output_file}")
        except Exception as e:
            log_event(f"🔥 Failed to save results to {output_file}: {e}", "ERROR")

if __name__ == "__main__":
    
    # This script runs from the 'evaluation/' directory
    # We need to make sure paths are correct
    
    # Path to test queries (inside 'evaluation/' folder)
    queries_file_path = 'test_queries.json' 
    # Path to save results (inside 'evaluation/results/' folder)
    results_file_path = 'results/metrics.json'
    
    evaluator = SystemEvaluator()

    try:
        results = evaluator.run_full_evaluation(queries_file_path)
        
        if results:
            print_header("EVALUATION COMPLETE")
            print(f"\n📊 Final Results:")
            print(f"  ├─ Retrieval Accuracy: {results.get('retrieval_accuracy', 0.0):.1f}%")
            print(f"  ├─ Hallucination Rate: {results.get('hallucination_rate', 0.0):.1f}%")
            print(f"  ├─ Average Confidence: {results.get('avg_confidence', 0.0):.2f}")
            print(f"  └─ Interaction Detection: {results.get('interaction_detection_precision', 0.0):.1f}%")
            
            evaluator.save_results(results_file_path)
        else:
            print("🔥 Evaluation did not produce results.")
            
    except Exception as e:
        log_event(f"Evaluation failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
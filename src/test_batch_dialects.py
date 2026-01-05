
import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.inference import GujaratiDialectPredictor

def run_batch_test():
    print("Loading Models for Batch Testing...")
    predictor = GujaratiDialectPredictor()
    
    test_cases = [
        # Standard Gujarati (Formal, Documentary style)
        {
            "text": "ગુજરાત રાજ્ય ભારતના પશ્ચિમ કિનારે આવેલું છે.", 
            "label": "Standard",
            "desc": "Formal Fact"
        },
        {
            "text": "ગાંધીજીએ સત્ય અને અহিংসાનો માર્ગ અપનાવ્યો હતો.", 
            "label": "Standard",
            "desc": "Gandhi History"
        },
        
        # Saurashtra / Kathiawadi (Folklore, Dayro, Colloquial)
        {
            "text": "રામ રામ સીતારામ! આવો બાપા આવો, પધારો... અટાણે ક્યાં નીકળ્યા’તા?", 
            "label": "Saurashtra/Kathiawadi",
            "desc": "User Example (Greetings)"
        },
        {
            "text": "હાલો પેલા ટાઢું પાણી પીવો અને પછી ગરમાગરમ રોટલો ખાવાનો છે.", 
            "label": "Saurashtra/Kathiawadi",
            "desc": "User Example (Food)"
        },
        {
            "text": "એલા ભાઈ, આ તો કાઠિયાવાડ છે હો, અહીયા મહેમાન ભગવાન કેવાય.", 
            "label": "Kathiawadi",
            "desc": "Regional Pride"
        },
        {
            "text": "કાં બાપા, સીદ જાવ છો? બળદિયા થાકી ગ્યા લાગે છે.", 
            "label": "Kathiawadi",
            "desc": "Rural Question"
        }
    ]
    
    print(f"\nRunning {len(test_cases)} Tests...\n")
    print(f"{'TYPE':<20} | {'PREDICTION':<15} | {'CONFIDENCE':<10} | {'TEXT (Truncated)'}")
    print("-" * 80)
    
    for case in test_cases:
        result = predictor.predict(text=case['text'])
        
        pred_label = result.get('final_prediction', 'N/A')
        print(f"{case['label']:<20} | {pred_label:<15} | {result.get('confidence',0):.4f}     | {case['text'][:30]}...")

if __name__ == "__main__":
    run_batch_test()

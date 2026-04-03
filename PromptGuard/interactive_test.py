from inference_pipeline import analyze_prompt
import sys

print("=============================================")
print("      PromptGuard - Interactive Tester       ")
print("=============================================\n")
print("Type a prompt and press Enter to scan it.")
print("Type 'exit' or 'quit' to close the tester.\n")

while True:
    try:
        user_input = input("Enter a prompt: ")
        
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue
            
        print("\nAnalyzing...")
        result = analyze_prompt(user_input)
        
        print("\n--- Result ---")
        print(f"Decision:          {result['decision']}")
        print(f"Final Risk Score:  {result['final_score']:.4f}")
        print(f"Model Probability: {result['model_probability']:.4f}")
        print(f"Rule Score:        {result['rule_score']:.4f}")
        print("----------------\n")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")

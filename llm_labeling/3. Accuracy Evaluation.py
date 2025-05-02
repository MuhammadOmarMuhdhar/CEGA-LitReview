import google.generativeai as genai
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Put API key here
API_KEY = " " # Put the API Key Here

# Configure the API
genai.configure(api_key=API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Read the CSV file and get the abstract column
df_filtered['predicted_context'] = ''  # Create new column for predictions

# Define different prompt templates
PROMPT_TEMPLATES = {
    "Basic": """Classify the following text into one of the four categories related to poverty context:
Low resource Level: Lack of financial or material resources, either absolute, perceived, or relative.
Resource volatility: Unpredictable or sudden changes in resources (e.g., job loss, financial instability).
Physical Environment: Conditions related to violence, noise, pollution, or neighborhood quality.
Social Environment: Social stigma, cultural influences, discrimination, norms, and stereotypes.
Return only the category name-Resource volatility, Low resource level, Social, Physical. If uncertain, predict by likelihood.
Text: {text}""",

    "Detailed": """As an expert in poverty research, analyze the following text and classify it into one of these categories:

1. Low resource Level
- Definition: Lack of financial or material resources (absolute, perceived, or relative)
- Examples: Income below poverty line, inability to afford basic needs, perceived financial hardship

2. Resource volatility
- Definition: Unpredictable or sudden changes in resources
- Examples: Job loss, financial instability, irregular income, unexpected expenses

3. Physical Environment
- Definition: Conditions related to living environment
- Examples: Violence, noise, pollution, poor neighborhood quality, inadequate housing

4. Social Environment
- Definition: Social and cultural factors
- Examples: Social stigma, discrimination, cultural barriers, negative stereotypes

Return only the category name (Resource volatility, Low resource level, Social, Physical).
Text: {text}""",

    "Structured": """[TASK] Classify poverty context from text
[CATEGORIES]
1. Low resource Level (lack of financial/material resources)
2. Resource volatility (unpredictable resource changes)
3. Physical Environment (environmental conditions)
4. Social Environment (social/cultural factors)

[INSTRUCTIONS]
- Analyze the text below
- Choose the most appropriate category
- Return only the category name
- Use exact names: Resource volatility, Low resource level, Social, Physical

[TEXT]
{text}"""
}

# Add retry decorator for handling rate limits
@retry(
    wait=wait_exponential(multiplier=3, min=5, max=120),
    stop=stop_after_attempt(10)
)
def get_model_prediction(text, prompt_template="Basic"):
    """Get model prediction using specified prompt template"""
    prompt = PROMPT_TEMPLATES[prompt_template].format(text=text)
    response = model.generate_content([text, prompt])
    return response.text.strip()

# Show initial dataset information and filtering
print(f"Initial dataset size: {len(df)} rows")
df_filtered = df[~df["Poverty Context"].str.contains(",", regex=False)].copy()
df_filtered = df_filtered.reset_index(drop=True)
print(f"Remaining dataset size: {len(df_filtered)} rows")

# Create columns for each prompt template
for template_name in PROMPT_TEMPLATES.keys():
    df_filtered[f'predicted_context_{template_name}'] = ''

# Add progress tracking variables
total_abstracts = len(df_filtered)
start_time = time.time()

# Process each row with different prompts
for idx, row in df_filtered.iterrows():
    try:
        elapsed_time = time.time() - start_time
        avg_time_per_abstract = elapsed_time / (idx + 1) if idx > 0 else 0
        estimated_time_remaining = avg_time_per_abstract * (total_abstracts - (idx + 1))
        
        print(f"\nProcessing abstract {idx + 1}/{total_abstracts} ({((idx + 1)/total_abstracts*100):.1f}%)")
        print(f"Time elapsed: {elapsed_time:.1f}s | Est. time remaining: {estimated_time_remaining:.1f}s")
        
        time.sleep(1)
        text = row['Abstract']
        
        for template_name in PROMPT_TEMPLATES.keys():
            print(f"  Using {template_name} prompt...", end=' ', flush=True)
            prediction = get_model_prediction(text, template_name)
            df_filtered.at[idx, f'predicted_context_{template_name}'] = prediction
            print(f"Predicted: {prediction}")
            
        if (idx + 1) % 10 == 0:
            print("\nIntermediate Results:")
            for template_name in PROMPT_TEMPLATES.keys():
                correct = (df_filtered[f'predicted_context_{template_name}'][:idx+1] == df_filtered['Poverty Context'][:idx+1]).sum()
                current_accuracy = (correct / (idx + 1)) * 100
                print(f"{template_name} Prompt Current Accuracy: {current_accuracy:.2f}%")
            print("-" * 50)
            
    except Exception as e:
        print(f"\nError processing abstract {idx}: {str(e)}")
        for template_name in PROMPT_TEMPLATES.keys():
            df_filtered.at[idx, f'predicted_context_{template_name}'] = 'Error'

# Calculate and display final results
print("\nAccuracy Comparison:")
for template_name in PROMPT_TEMPLATES.keys():
    correct_predictions = (df_filtered[f'predicted_context_{template_name}'] == df_filtered['Poverty Context']).sum()
    accuracy = (correct_predictions / len(df_filtered)) * 100
    print(f"{template_name} Prompt Accuracy: {accuracy:.2f}%")

# Save results
results_df = df_filtered[['Abstract', 'Poverty Context'] + 
                        [f'predicted_context_{template}' for template in PROMPT_TEMPLATES.keys()]]
results_df.to_csv('prompt_comparison_results.csv', index=False)
print("\nDetailed results saved to 'prompt_comparison_results.csv'")

# Display confusion matrices and classification reports for each template
for template_name in PROMPT_TEMPLATES.keys():
    print(f"\nConfusion Matrix for {template_name} Prompt:")
    print(confusion_matrix(df_filtered['Poverty Context'], 
                         df_filtered[f'predicted_context_{template_name}']))
    print(f"\nClassification Report for {template_name} Prompt:")
    print(classification_report(df_filtered['Poverty Context'], 
                              df_filtered[f'predicted_context_{template_name}']))

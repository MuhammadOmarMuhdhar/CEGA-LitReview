import pandas as pd
import time
import google.generativeai as genai

# Set the Gemini API key
genai.configure(api_key="AIzaSyADXw6c9rtAGXEWgImvqmmBZWCVQD28sN4")

# Load CSV file (the file storing the web scraped papers)
df = pd.read_csv(“XXXXXX.csv")     # Remember to change the directory and name

# Ensure the CSV contains an "abstract" column
if "abstract" not in df.columns:
    raise ValueError("CSV file must contain an 'abstract' column.")

# Define batch classification function
def classify_abstracts_in_batch(abstracts):
    prompt = "Classify each abstract as 'Relevant' or 'Not Relevant' to the topic of Economics and Psychology of Poverty.\n\n"
    
    for i, abstract in enumerate(abstracts):
        prompt += f"Abstract {i+1}: {abstract}\n"

    prompt += "\nRespond with a list of labels, one per line, in the same order."

    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content(prompt)
        time.sleep(2)  # Prevent hitting API limits
        results = response.text.strip().split("\n")  # Assuming responses are newline-separated
        return results if len(results) == len(abstracts) else ["Uncertain"] * len(abstracts)
    except Exception as e:
        print(f"Error processing abstracts: {e}")
        return ["Error"] * len(abstracts)

# Process abstracts in batches of 10 to minimize API calls
batch_size = 10
df["relevance"] = "Pending"

for i in range(0, len(df), batch_size):
    batch = df.iloc[i : i + batch_size]["abstract"].tolist()
    results = classify_abstracts_in_batch(batch)
    df.loc[i : i + batch_size - 1, "relevance"] = results

# Save the results
df.to_csv("classified_abstracts.csv", index=False)

print("Classification complete! Results saved in 'classified_abstracts.csv'.")
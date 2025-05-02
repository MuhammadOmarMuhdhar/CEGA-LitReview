import pandas as pd
import time
import google.generativeai as genai

# === Load Data ===
df = pd.read_excel("full_sample_dataset.xlsx") # Replace with the path to the dataset
df["abstract"] = df["abstract"].fillna("")
df = df[df["abstract"].str.strip() != ""]  # remove empty abstracts

# === Zero-Shot Classifier ===
class ZeroShotSufficiencyClassifier:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.last_request_time = 0
        self.min_request_interval = 4.0

    def _wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def predict(self, abstract: str) -> int:
        self._wait()
        prompt = f"""
Answer with 1 or 0 only.

Does the abstract below provide information about **research study type**?

Abstract: {abstract}

Output 1 if the abstract is sufficient, otherwise output 0. Only return the number.
"""
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            return 1 if "1" in answer and "0" not in answer else 0
        except Exception as e:
            print("Error:", e)
            return 0

# === Run 3 Iterations ===
API_KEY = " " # the API key for Gemini
classifier = ZeroShotSufficiencyClassifier(API_KEY)

n_samples = 100
n_iterations = 5
percentages = []

for i in range(n_iterations):
    print(f"\n--- Iteration {i+1} ---")
    sample = df.sample(n=n_samples, random_state=42 + i)["abstract"].tolist()
    results = [classifier.predict(text) for text in sample]
    sufficient_count = sum(results)
    percentage = sufficient_count / n_samples * 100
    percentages.append(percentage)
    print(f"Sufficient: {sufficient_count}/{n_samples} → {percentage:.2f}%")

print("\n=== Summary ===")
for i, p in enumerate(percentages, 1):
    print(f"Iteration {i}: {p:.2f}% sufficient")
print(f"Average across {n_iterations} iterations: {sum(percentages)/len(percentages):.2f}%")
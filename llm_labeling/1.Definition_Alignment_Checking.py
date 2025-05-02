import time
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

# Initialize embedding model (can be any SBERT model)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class GeminiContextComparer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.min_request_interval = 4.0
        self.last_request_time = 0

    def _wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def query_gemini(self, prompt):
        self._wait()
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def compare_understanding(self, label, definition, example_question=None):
        # Query with label only (zero-shot)
        prompt_zero_shot = f"What does the concept '{label}' mean?"
        zero_shot_response = self.query_gemini(prompt_zero_shot)

        # Query with label and user definition
        prompt_with_context = f"Consider the definition: '{label}' means \"{definition}\".\nNow, explain what '{label}' means."
        contextual_response = self.query_gemini(prompt_with_context)

        # Compare the two explanations
        embeddings = embedding_model.encode([zero_shot_response, contextual_response], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        return {
            "label": label,
            "definition": definition,
            "gemini_zero_shot": zero_shot_response,
            "gemini_with_definition": contextual_response,
            "similarity_score": similarity
        }
    
comparer = GeminiContextComparer(api_key="")# To use, please put the API Key Here
result = comparer.compare_understanding(
    label="Low Resource Level",
    definition="Difficulty in meeting basic needs"
)
print(result)

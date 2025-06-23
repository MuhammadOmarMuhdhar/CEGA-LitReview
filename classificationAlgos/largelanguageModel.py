import time
import google.generativeai as genai
import json
import logging
import pycountry
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import tiktoken
# from collections import defaultdict # Not used, can be removed
# import pandas as pd # Not used, can be removed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeminiModel:
    def __init__(self, api_keys, max_abstracts_per_batch=5, max_input_tokens=2000, max_output_tokens=1000, delay=2, model_name="gemma-3-27b-it", temperature=0.3,
                 retry_model_name='gemini-2.0-flash', retry_max_input_tokens=10000, retry_max_output_tokens=1000, retry_delay=5, retry_temperature=None, max_retries=10, 
                 retry_model_max_abstracts_per_batch=10):
        """
        Initializes the GeminiModel for parallel abstract classification with numeric optimizations.

        Args:
            api_keys: List of API keys or a single API key string.
            max_input_tokens: Maximum input tokens allowed per batch for initial attempts.
            max_output_tokens: Maximum output tokens allowed per batch for initial attempts.
            delay: Delay in seconds between API calls per key to manage rate limiting for initial attempts.
            model_name: Gemini model name to use for initial attempts (e.g., "gemini-2.0-flash").
            temperature: Controls the randomness of the model's output for initial attempts. Lower values (e.g., 0.1-0.3)
                         result in more deterministic output.
            retry_model_name: Model name to use for retries. If None, uses the initial model_name.
            retry_max_input_tokens: Max input tokens for retry batches. If None, uses initial max_input_tokens.
            retry_max_output_tokens: Max output tokens for retry batches. If None, uses initial max_output_tokens.
            retry_delay: Delay for retry API calls. If None, uses initial delay.
            retry_temperature: Temperature for retry API calls. If None, uses initial temperature.
            max_retries: Maximum number of retry attempts for a single API call.
        """
        # Ensure api_keys is a list
        self.api_keys = [api_keys] if isinstance(api_keys, str) else api_keys
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_abstracts_per_batch = max_abstracts_per_batch
        self.model_name = model_name
        self.delay = delay
        self.temperature = temperature
        self.max_retries = max_retries # Max retries for _generate method

        # Retry-specific configurations
        self.retry_model_name = retry_model_name if retry_model_name is not None else self.model_name
        self.retry_max_input_tokens = retry_max_input_tokens if retry_max_input_tokens is not None else self.max_input_tokens
        self.retry_max_output_tokens = retry_max_output_tokens if retry_max_output_tokens is not None else self.max_output_tokens
        self.retry_delay = retry_delay if retry_delay is not None else self.delay
        self.retry_temperature = retry_temperature if retry_temperature is not None else self.temperature
        self.retry_model_max_abstracts_per_batch = retry_model_max_abstracts_per_batch if retry_model_max_abstracts_per_batch is not None else self.max_abstracts_per_batch


        # Initialize tiktoken encoder
        self.tokenizer = tiktoken.get_encoding("o200k_base")

        # Initialize caches and GenerativeModel instances for each API key
        self.caches = {key: {} for key in self.api_keys}
        # Initialize models for both initial and retry configurations
        self.models = {}
        for api_key in self.api_keys:
            self.models[api_key] = {
                'initial': self.model_name,
                'retry': self.retry_model_name
            }
        # Thread locks for each API key to manage per-key rate limiting
        self.locks = {key: threading.Lock() for key in self.api_keys}
        # Shared rate limit tracking across threads
        self.api_key_cooldowns = {key: 0 for key in self.api_keys} # Track cooldown end times
        self.cooldown_lock = threading.Lock()
        # Initialize label mappings; these will be populated by _create_label_mappings
        self.label_mappings = {}
        self.reverse_mappings = {}
        # Output field mappings - converts field names to numbers
        self.field_mappings = {
            'study_type': 0,
            'poverty_context': 1,
            'mechanism': 2,
            'behavior': 3
        }
        self.reverse_field_mappings = {v: k for k, v in self.field_mappings.items()}
        # Pre-calculate token overhead for the static part of the prompt
        self.base_prompt_tokens = self._calculate_base_prompt_tokens()
        logger.info(f"Base prompt token overhead: {self.base_prompt_tokens} tokens")
        logger.info(f"Initialized with {len(self.api_keys)} API keys for parallel processing")

    def _calculate_base_prompt_tokens(self) -> int:
        """
        Calculates the token count of the static part of the prompt using tiktoken.
        This helps in determining how many abstract tokens can fit into a batch.
        """
        # Generate a prompt with an empty abstracts_batch to get the base prompt size
        # This includes all instructions and label mappings but no abstract content.
        dummy_abstracts_batch = {}
        # Dummy data for prompt creation to ensure all static parts are counted
        study_types = ["Type A"]
        poverty_contexts = ["Context X"]
        mechanisms = ["Mechanism M"]
        behaviors = ["Behavior P"]

        base_prompt = self._create_prompt(
            dummy_abstracts_batch, study_types, poverty_contexts, mechanisms, behaviors
        )
        return len(self.tokenizer.encode(base_prompt))

    def _count_tokens(self, text: str) -> int:
        """
        Counts tokens for a given text using tiktoken.
        """
        return len(self.tokenizer.encode(text))

    def _estimate_output_tokens(self, num_abstracts: int) -> int:
        """
        Estimates the output token count for a given number of abstracts.
        Based on the optimized JSON structure with numeric keys and DOI mappings.
        """
        # Optimized structure: {"0": {"0": [999], "1": [999], "2": [999], "3": [999]}}
        # Numeric DOI (1-3 digits) + numeric field keys (1 digit) + values (1-3 digits each)
        # Plus JSON formatting characters
        tokens_per_abstract = 20
        # Array wrapper tokens
        array_overhead = 10
        return (num_abstracts * tokens_per_abstract) + array_overhead

    def _create_label_mappings(self, study_types: list, poverty_contexts: list, mechanisms: list, behaviors: list):
        """
        Creates numeric mappings for all label categories.
        These mappings are used in the prompt and for decoding results.
        """
        self.label_mappings = {
            'study_type': {label: i for i, label in enumerate(study_types)},
            'poverty_context': {label: i for i, label in enumerate(poverty_contexts)},
            'mechanism': {label: i for i, label in enumerate(mechanisms)},
            'behavior': {label: i for i, label in enumerate(behaviors)}
        }
        # Add a special code for "Insufficient info" for all categories
        insufficient_code = 999
        for category in self.label_mappings:
            self.label_mappings[category]["Insufficient info"] = insufficient_code
        # Create reverse mappings for decoding numeric codes back to text labels
        self.reverse_mappings = {
            category: {num: label for label, num in mapping.items()}
            for category, mapping in self.label_mappings.items()
        }

    def _create_doi_mapping(self, abstracts_batch: dict) -> tuple[dict, dict]:
        """
        Creates numeric mapping for DOIs in current batch to save tokens.
        """
        doi_to_num = {doi: i for i, doi in enumerate(abstracts_batch.keys())}
        num_to_doi = {i: doi for doi, i in doi_to_num.items()}
        return doi_to_num, num_to_doi
    
    def _get_available_api_keys(self) -> list:
        """
        Returns list of API keys not currently in cooldown.
        """
        with self.cooldown_lock:
            current_time = time.time()
            return [key for key in self.api_keys if current_time >= self.api_key_cooldowns[key]]

    def _generate(self, prompt: str, api_key: str, attempt_type: str = 'initial') -> str:
        """
        Generates content using a specific API key with comprehensive retry logic.
        Uses model and parameters based on `attempt_type` ('initial' or 'retry').
        """
        cache = self.caches[api_key]
        if prompt in cache:
            return cache[prompt]

        # Determine which model and parameters to use based on attempt_type
        model_name = self.models[api_key][attempt_type]
        current_delay = self.delay if attempt_type == 'initial' else self.retry_delay
        current_temperature = self.temperature if attempt_type == 'initial' else self.retry_temperature
        current_max_retries = self.max_retries

        # Use lock to ensure rate limiting per API key AND fix race condition
        with self.locks[api_key]:
            # Check cooldown INSIDE the API key lock to prevent race conditions
            with self.cooldown_lock:
                current_time = time.time()
                if current_time < self.api_key_cooldowns[api_key]:
                    cooldown_remaining = self.api_key_cooldowns[api_key] - current_time
                    logger.info(f"API key {api_key[:6]}... is in cooldown. Waiting for {cooldown_remaining:.2f} seconds.")
                    time.sleep(cooldown_remaining)

            # Configure API key for this specific request (fixes the global configuration issue)
            genai.configure(api_key=api_key)
            
            # Create model instance with the correct configuration
            model_instance = genai.GenerativeModel(model_name)
            
            for attempt in range(current_max_retries):
                try:
                    time.sleep(current_delay)  # Rate limit management
                    generation_config = {
                        "temperature": current_temperature,
                    }
                    response = model_instance.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    output = response.text
                    cache[prompt] = output
                    return output
                    
                except Exception as e:
                    # More robust error handling for different types of rate limiting
                    is_rate_limited = False
                    retry_seconds = 10  # Default fallback
                    
                    # Check for rate limiting in various ways the API might indicate it
                    if hasattr(e, 'status_code') and e.status_code == 429:
                        is_rate_limited = True
                    elif hasattr(e, 'retry_delay'):
                        is_rate_limited = True
                        if hasattr(e.retry_delay, 'seconds'):
                            retry_seconds = e.retry_delay.seconds
                        else:
                            retry_seconds = e.retry_delay
                    elif 'rate limit' in str(e).lower() or 'quota' in str(e).lower():
                        is_rate_limited = True
                    
                    if is_rate_limited:
                        logger.warning(f"Rate limited on attempt {attempt + 1} ({attempt_type}) with API key {api_key[:6]}... Waiting {retry_seconds} seconds...")
                        with self.cooldown_lock:
                            self.api_key_cooldowns[api_key] = time.time() + retry_seconds
                        time.sleep(retry_seconds)
                    else:
                        logger.error(f"Error on attempt {attempt + 1} ({attempt_type}) with API key {api_key[:6]}...: {e}")
                    
                    if attempt == current_max_retries - 1:
                        raise e  # Re-raise if all retries are exhausted
                
    def _create_prompt(self, abstracts_batch: dict, study_types: list, poverty_contexts: list, mechanisms: list, behaviors: list) -> str:
        """
        Creates a prompt for the Gemini model to process a batch of abstracts,
        including instructions and numeric mappings for both DOIs and field keys.
        """
        doi_to_num, num_to_doi = self._create_doi_mapping(abstracts_batch)

        formatted_abstracts = ""
        for doi, abstract in abstracts_batch.items():
            numeric_id = doi_to_num[doi]
            formatted_abstracts += f"ID: {numeric_id}\nAbstract: {abstract}\n\n"

        study_type_mapping = "\n".join([f"{i}: {label}" for i, label in enumerate(study_types)])
        poverty_context_mapping = "\n".join([f"{i}: {label}" for i, label in enumerate(poverty_contexts)])
        mechanism_mapping = "\n".join([f"{i}: {label}" for i, label in enumerate(mechanisms)])
        behavior_mapping = "\n".join([f"{i}: {label}" for i, label in enumerate(behaviors)])

        prompt = f"""
You will analyze {len(abstracts_batch)} research abstracts. For each abstract, assign numeric codes in the following four categories:

**Field Key Mappings:**
0: Study Type (can assign multiple numbers if applicable)
1: Poverty Context (can assign multiple numbers if applicable)
2: Mechanism (can assign multiple numbers if applicable)
3: Behavior (can assign multiple numbers if applicable)

**Critical Instructions:**
- You must assign **at least one** numeric code for **each** of the four fields (0–3).
- Use ONLY the numeric codes provided below for all categories.
- Assign multiple codes if the abstract supports more than one. Format as: `[1, 3, 5]`. A single label may be written as `2` or `[2]`.
- Do **not** leave any field blank. Return a full set of labels for every abstract.

**On Using '999' (Insufficient Info):**
- Use code `999` **only if no reasonable inference is possible** after full consideration of the abstract and plausible background knowledge.
- You are expected to **infer beyond the literal abstract** where appropriate, based on:
    - Field knowledge
    - The study setting or goals
    - Known practices in similar research

**Labeling Guidance by Category:**
- **Study Type:** Infer from references to interventions, design, or data (e.g., control group implies experiment).
- **Poverty Context:** Infer from setting, economic indicators, demographic focus, or target population.
- **Mechanisms:** Identify cognitive, psychological, or behavioral processes mentioned or implied.
- **Behavior:** Identify actions, decisions, or patterns studied or influenced.

**Numeric Label Codes:**

**Study Type Codes (Field 0):** {study_type_mapping}
999: Insufficient info

**Poverty Context Codes (Field 1):** {poverty_context_mapping}
999: Insufficient info

**Mechanism Codes (Field 2):** {mechanism_mapping}
999: Insufficient info

**Behavior Codes (Field 3):** {behavior_mapping}
999: Insufficient info

**Input Abstracts:** {formatted_abstracts}

**IMPORTANT:** Return a valid JSON array. Each entry must include all four fields (0, 1, 2, 3), with one or more numeric codes per field.

**Output Format Example:**
[
{{"0": [0, 2], "1": [1], "2": [2, 4], "3": [3]}},
{{"0": [1], "1": [0], "2": [5], "3": [2]}}
]
"""
        return prompt

    def _parse_json_response(self, json_str_to_parse: str, api_key_snippet: str, doi_mapping: dict) -> dict:
        """
        Robustly parses a JSON string response from the Gemini model, handling the array format
        and converting numeric DOI/field keys back to original format.
        Uses simplified JSON extraction approach.
        """
        numeric_results = {}
        parsed_json = None
        try:
            # Try direct JSON parsing first
            parsed_json = json.loads(json_str_to_parse)
            logger.info("Successfully parsed JSON directly.")
        except json.JSONDecodeError as e:
            # Use the simpler extraction method from the second version
            start_idx = json_str_to_parse.find('{')
            if json_str_to_parse.find('[') >= 0 and (start_idx == -1 or json_str_to_parse.find('[') < start_idx):
                # Array format detected
                start_idx = json_str_to_parse.find('[')
                end_idx = json_str_to_parse.rfind(']') + 1
            else:
                # Object format
                end_idx = json_str_to_parse.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str_extracted = json_str_to_parse[start_idx:end_idx]
                try:
                    parsed_json = json.loads(json_str_extracted)
                    logger.info("Successfully parsed JSON after extraction.")
                except json.JSONDecodeError as e_extract:
                    raise ValueError(f"Could not extract valid JSON from response for API key {api_key_snippet}: {e_extract}")
            else:
                raise ValueError(f"Could not find valid JSON object in response for API key {api_key_snippet}")

        # Convert array format to flat dictionary or handle nested format
        if isinstance(parsed_json, list):
            # Handle array format where each item corresponds to a DOI by index
            for idx, fields in enumerate(parsed_json):
                if isinstance(fields, dict):
                    numeric_results[str(idx)] = fields
        elif isinstance(parsed_json, dict):
            # Handle nested format directly
            numeric_results = parsed_json
        else:
            raise ValueError(f"Unexpected JSON structure: {type(parsed_json)}")

        # Convert numeric DOI and field keys back to original format
        converted_results = {}
        for numeric_doi, fields in numeric_results.items():
            # Convert numeric DOI back to original DOI
            try:
                numeric_doi_int = int(numeric_doi)
                original_doi = doi_mapping[numeric_doi_int]
            except (ValueError, KeyError):
                logger.warning(f"Could not map numeric DOI {numeric_doi} back to original DOI")
                continue

            # Convert numeric field keys back to original field names
            converted_fields = {}
            for numeric_field, values in fields.items():
                try:
                    numeric_field_int = int(numeric_field)
                    original_field = self.reverse_field_mappings[numeric_field_int]
                    converted_fields[original_field] = values
                except (ValueError, KeyError):
                    logger.warning(f"Could not map numeric field {numeric_field} back to original field name")
                    continue
            converted_results[original_doi] = converted_fields
        return converted_results

    def _decode_numeric_labels(self, numeric_results: dict) -> tuple[dict, dict]:
        """
        Converts numeric codes in the classification results back to text labels.
        Handles both single numeric values and lists of numeric values.
        Returns decoded results and a dictionary of papers with invalid codes.
        """
        decoded_results = {}
        invalid_papers = {} # Track papers with invalid codes
        for doi, labels in numeric_results.items():
            decoded_labels = {}
            has_invalid_code = False
            # Decode each category
            for category in ['study_type', 'poverty_context', 'mechanism', 'behavior']:
                if category in labels:
                    numeric_codes = labels[category]
                    # Ensure numeric_codes is iterable (handle single int vs list)
                    code_list = [numeric_codes] if not isinstance(numeric_codes, list) else numeric_codes
                    # Decode each numeric code to text
                    decoded_texts = []
                    for code in code_list:
                        if code in self.reverse_mappings[category]:
                            decoded_texts.append(self.reverse_mappings[category][code])
                        else:
                            logger.warning(f"Invalid numeric code {code} for category {category} in DOI {doi}")
                            has_invalid_code = True
                            break # Break from inner loop on first invalid code
                    if has_invalid_code:
                        break # Break from outer loop if an invalid code was found
                    # Join multiple labels with comma or return single label
                    decoded_labels[category] = ", ".join(decoded_texts) if len(decoded_texts) > 1 else (decoded_texts[0] if decoded_texts else "Insufficient info")
                else:
                    # If category is missing in the model's response, mark as insufficient
                    decoded_labels[category] = "Insufficient info"

            if has_invalid_code:
                invalid_papers[doi] = labels # Store original numeric labels for failed papers
            else:
                decoded_results[doi] = decoded_labels
        return decoded_results, invalid_papers

    def _process_batch_with_key(self, batch_data: tuple, attempt_type: str = 'initial') -> tuple[dict, dict]:
        """
        Processes a single batch of abstracts using an assigned API key.
        Handles API calls, JSON parsing, and decoding of numeric labels.
        Returns successfully decoded results and any abstracts that failed.
        `attempt_type` specifies if it's an 'initial' call or a 'retry'.
        """
        batch, api_key, study_types, poverty_contexts, mechanisms, behaviors = batch_data
        try:
            # Create DOI mapping for this batch
            doi_to_num, num_to_doi = self._create_doi_mapping(batch)
            # Generate classification response from the model
            batch_raw_response = self._generate(
                self._create_prompt(batch, study_types, poverty_contexts, mechanisms, behaviors),
                api_key,
                attempt_type=attempt_type # Pass the attempt type
            )
            logger.info(f"Successfully classified batch of {len(batch)} abstracts using API key: {api_key[:6]}... ({attempt_type} attempt)")
            # Robustly parse the JSON response with DOI mapping
            numeric_results = self._parse_json_response(batch_raw_response, api_key + '...', num_to_doi)
            # Decode numeric codes back to text labels
            decoded_results, invalid_papers = self._decode_numeric_labels(numeric_results)
            # Determine which papers from the current batch successfully processed
            failed_abstracts = {doi: batch[doi] for doi in batch if doi not in decoded_results}
            return decoded_results, failed_abstracts
        except Exception as e:
            logger.error(f"Error processing batch with API key {api_key[:6]}... ({attempt_type} attempt): {e}")
            return {}, batch # If any error, assume all papers in batch failed

    def _create_abstract_batches(self, all_abstracts: dict, categories: tuple, max_input_tokens: int, max_output_tokens: int, max_abstracts_per_batch: int = None) -> list:
        """
        Generates batches of abstracts, ensuring each batch respects both input and output token limits.
        Assigns API keys in a round-robin fashion to each batch.
        Now accounts for optimized token usage with numeric DOIs and field keys.

        Args:
            all_abstracts: Dictionary of all abstracts {doi: abstract_text}.
            categories: Tuple of (study_types, poverty_contexts, mechanisms, behaviors).
            max_input_tokens: The maximum input token count allowed per batch.
            max_output_tokens: The maximum output token count allowed per batch.
            max_abstracts_per_batch: Optional, maximum number of abstracts per batch.

        Returns:
            A list of tuples, where each tuple is (abstracts_batch_dict, api_key, *categories).
        """
        study_types, poverty_contexts, mechanisms, behaviors = categories
        batches = []
        current_batch_abstracts = {}
        current_batch_input_tokens = self.base_prompt_tokens # Start with the prompt's fixed overhead
        abstracts_items = list(all_abstracts.items())

        for doi, abstract in abstracts_items:
            # For token counting, use optimized format (numeric ID instead of full DOI)
            numeric_id = len(current_batch_abstracts) # Approximate numeric ID
            abstract_content_for_prompt = f"ID: {numeric_id}\nAbstract: {abstract}\n\n"
            abstract_tokens = self._count_tokens(abstract_content_for_prompt)

            # Estimate output tokens for current batch size + 1
            estimated_output_tokens = self._estimate_output_tokens(len(current_batch_abstracts) + 1)

            # Calculate new input tokens
            new_input_tokens = current_batch_input_tokens + abstract_tokens

            # Check if adding this abstract exceeds either input or output token limits, or abstract count limit
            if (new_input_tokens > max_input_tokens or
                estimated_output_tokens > max_output_tokens or
                (self.max_abstracts_per_batch and len(current_batch_abstracts) >= self.max_abstracts_per_batch)):
                if current_batch_abstracts: # Only add if the batch is not empty
                    api_key = self.api_keys[len(batches) % len(self.api_keys)] # Round-robin API key
                    batches.append((current_batch_abstracts, api_key, *categories))
                # Start a new batch with the current abstract
                current_batch_abstracts = {doi: abstract}
                current_batch_input_tokens = self.base_prompt_tokens + abstract_tokens
            else:
                # Add abstract to the current batch
                current_batch_abstracts[doi] = abstract
                current_batch_input_tokens = new_input_tokens

        # Add any remaining abstracts in the last batch
        if current_batch_abstracts:
            available_keys = self._get_available_api_keys()
            if not available_keys:
                available_keys = self.api_keys # Fallback to all keys if none available
            api_key = available_keys[len(batches) % len(available_keys)]
            batches.append((current_batch_abstracts, api_key, *categories))

        return batches

    def _retry_failed_papers(self, failed_abstracts: dict, categories: tuple, max_retry_attempts: int = 3) -> dict:
        """
        Retries processing abstracts that failed initial classification.
        Uses retry-specific model, token limits, and delay.
        """
        successfully_processed_retries = {}
        retries_remaining = max_retry_attempts

        while failed_abstracts and retries_remaining > 0:
            retries_remaining -= 1
            logger.info(f"Retry attempt for {len(failed_abstracts)} papers. {retries_remaining} attempts remaining.")

            # Create retry batches using retry-specific configurations
            retry_batches = self._create_abstract_batches(
                failed_abstracts, categories, self.retry_max_input_tokens, self.retry_max_output_tokens,
                max_abstracts_per_batch= self.retry_model_max_abstracts_per_batch 
            )
            logger.info(f"Created {len(retry_batches)} retry batches using retry configurations.")

            with ThreadPoolExecutor(max_workers=len(self.api_keys)) as executor:
                future_to_batch = {executor.submit(self._process_batch_with_key, batch_data, 'retry'): batch_data[0]
                                   for batch_data in retry_batches}

                for future in as_completed(future_to_batch):
                    try:
                        decoded_results, batch_failed_on_retry = future.result()
                        successfully_processed_retries.update(decoded_results)
                        # Remove successfully processed papers from the `failed_abstracts` list
                        for doi in decoded_results:
                            failed_abstracts.pop(doi, None)
                    except Exception as e:
                        logger.error(f"Error in retry batch: {e}")

            if failed_abstracts and retries_remaining > 0:
                # Exponential backoff between retry rounds
                wait_time = self.retry_delay * (max_retry_attempts - retries_remaining)
                logger.info(f"Waiting for {wait_time:.2f} seconds before next retry round.")
                time.sleep(wait_time)

        return successfully_processed_retries

    def _validate_and_fix_labels(self, labels_dict: dict, doi: str) -> dict:
        """
        Validates that all required classification fields are present for a given DOI
        and fills in "Insufficient info" for any missing fields.
        """
        required_fields = ['study_type', 'poverty_context', 'mechanism', 'behavior']
        if doi not in labels_dict:
            logger.warning(f"DOI {doi} not found in classification results, providing default 'Insufficient info' for all fields.")
            return {field: "Insufficient info" for field in required_fields}
        result = labels_dict[doi]
        # Check if all required fields are present and fix missing ones
        for field in required_fields:
            if field not in result or result[field] is None or (isinstance(result[field], str) and result[field].strip() == ""):
                logger.warning(f"Missing or empty field '{field}' for DOI {doi}, setting to 'Insufficient info'")
                result[field] = "Insufficient info"
        return result

    def classify(self, papers: list, study_types: list, poverty_contexts: list, mechanisms: list, behaviors: list, max_papers_to_classify: int = None) -> list:
        """
        Classifies a list of research abstracts using parallel processing with multiple API keys,
        batching based on token limits. Includes retry mechanism for failed classifications.
        Uses numeric optimizations for DOIs and field keys to save tokens.

        Args:
            papers: List of paper dictionaries, each containing 'doi' and 'abstract'.
            study_types: List of study type definitions.
            poverty_contexts: List of poverty context definitions.
            mechanisms: List of mechanism definitions.
            behaviors: List of behavior definitions.
            max_papers_to_classify: Optional. If provided, limits the classification to this many papers.
                                    The papers will be taken from the beginning of the `papers` list.
            max_abstracts_per_batch: Optional. If provided, limits the number of abstracts in each batch.

        Returns:
            List of paper dictionaries updated with classification results.
        """
        # Store categories for consistent access
        categories = (study_types, poverty_contexts, mechanisms, behaviors)

        # Create label mappings first, as they are needed for decoding
        self._create_label_mappings(*categories)

        # Filter valid abstracts from the input papers
        all_abstracts = {p['doi']: p['abstract'] for p in papers if 'doi' in p and 'abstract' in p and p['abstract']}
        if not all_abstracts:
            logger.warning("No valid abstracts found in input papers.")
            return papers

        # Apply the total classification limit if specified
        if max_papers_to_classify is not None and max_papers_to_classify > 0:
            original_abstracts_count = len(all_abstracts)
            # Create a new dictionary with only the limited number of abstracts
            limited_abstracts = {}
            for i, (doi, abstract) in enumerate(all_abstracts.items()):
                if i >= max_papers_to_classify:
                    break
                limited_abstracts[doi] = abstract
            all_abstracts = limited_abstracts
            if len(all_abstracts) < original_abstracts_count:
                logger.info(f"Limiting total classification to {len(all_abstracts)} papers out of {original_abstracts_count} available.")


        logger.info(f"Starting classification for {len(all_abstracts)} abstracts...")
        # Initial batch creation, now respecting max_abstracts_per_batch
        batches = self._create_abstract_batches(
            all_abstracts, categories, self.max_input_tokens, self.max_output_tokens, self.max_abstracts_per_batch # Use self.max_abstracts_per_batch for consistency if not overridden
        )
        logger.info(f"Created {len(batches)} initial batches for classification.")
        final_results = {}
        failed_abstracts_initial = {}

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=len(self.api_keys)) as executor:
            future_to_batch = {executor.submit(self._process_batch_with_key, batch_data, 'initial'): batch_data[0]
                               for batch_data in batches}
            for future in as_completed(future_to_batch):
                try:
                    decoded_results, batch_failed = future.result()
                    final_results.update(decoded_results)
                    failed_abstracts_initial.update(batch_failed) # Collect abstracts that failed this batch
                except Exception as e:
                    logger.error(f"Error in main batch processing: {e}")

        # Retry failed abstracts
        if failed_abstracts_initial:
            logger.info(f"Attempting to retry {len(failed_abstracts_initial)} failed abstracts.")
            successfully_retried = self._retry_failed_papers(failed_abstracts_initial, categories, max_retry_attempts=3)
            final_results.update(successfully_retried)

        # Prepare final output: update original papers with classification results
        classified_papers = []
        for p in papers:
            doi = p.get('doi')
            if doi and doi in all_abstracts: # Only process papers that were part of the initial classification set
                # Validate and fix labels for each paper
                validated_labels = self._validate_and_fix_labels(final_results, doi)
                p.update(validated_labels)
            else:
                # If a paper wasn't processed (e.g., no abstract or not in limited set),
                # ensure it still has the classification fields with "Insufficient info"
                for field in ['study_type', 'poverty_context', 'mechanism', 'behavior']:
                    if field not in p:
                        p[field] = "Insufficient info"
            classified_papers.append(p)

        return classified_papers
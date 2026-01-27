import pandas as pd
import time
import re

from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix


# ============================================================
# 🔑 API CONFIGURATION
# ============================================================

API_KEY = "YOUR_OPENAI_API_KEY_HERE"

client = OpenAI(api_key=API_KEY)
MODEL_NAME = "gpt-4o"  # or "gpt-4o-mini" for cheaper testing


# ============================================================
# 📁 FILE CONFIGURATION
# ============================================================

files_to_process = [
    {
        "input": "/users/hasancan/Downloads/TR_OPETs.csv",
        "output": "gpt4_TR_OPETs_results.csv",
        "lang": "Turkish",
    },
    {
        "input": "/users/hasancan/Downloads/TR_NOPETs_fixed.csv",
        "output": "gpt4_TR_NOPETs_results.csv",
        "lang": "Turkish",
    },
    {
        "input": "/users/hasancan/Downloads/EN_OPETs.csv",
        "output": "gpt4_EN_OPETs_results.csv",
        "lang": "English",
    },
    {
        "input": "/users/hasancan/Downloads/EN_NOPETs_fixed.csv",
        "output": "gpt4_EN_NOPETs_results.csv",
        "lang": "English",
    },
]

SENTENCE_COL = "text"
TERM_COL = "PET"
LABEL_COL = "label"


# ============================================================
# 📊 GLOBAL TRACKING VARIABLES
# ============================================================

total_cost = 0
refusal_count = 0
unexpected_count = 0


# ============================================================
# 🧹 PREPROCESSING
# ============================================================

def clean_sentence(sentence, term):
    """Remove PET boundary markers and normalize spacing."""
    clean = re.sub(r"\[PET_BOUNDARY\]", "", str(sentence))
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


# ============================================================
# 🤖 GPT-4 INFERENCE
# ============================================================

def classify_euphemism(term, sentence, language="English"):
    """
    Classify whether a term is used euphemistically or literally.

    Args:
        term (str): The PET to classify
        sentence (str): Sentence containing the PET
        language (str): "English" or "Turkish"

    Returns:
        str: "Euphemistic", "Literal", "Unexpected", "Refusal", or "Error"
    """
    global total_cost, refusal_count, unexpected_count

    if language == "Turkish":
        system_prompt = (
            "You are a linguistics expert analyzing Turkish text for euphemistic language. "
            "A euphemism is an indirect expression that softens harsh, offensive, or taboo "
            "concepts like death, illness, firing, bodily functions, or sexual activity. "
            'Respond with ONLY one word: "Euphemistic" or "Literal".'
        )

        user_prompt = f"""
Analyze this Turkish sentence:

Sentence: "{sentence}"
Term: "{term}"

Question: Is the term "{term}" used as a EUPHEMISM (indirect, softening expression)
or LITERALLY (direct meaning)?

Classification:
"""
    else:
        system_prompt = (
            "You are a linguistics expert analyzing English text for euphemistic language. "
            "A euphemism is an indirect expression that softens harsh, offensive, or taboo "
            "concepts like death, illness, firing, bodily functions, or sexual activity. "
            'Respond with ONLY one word: "Euphemistic" or "Literal".'
        )

        user_prompt = f"""
Analyze this English sentence:

Sentence: "{sentence}"
Term: "{term}"

Question: Is the term "{term}" used as a EUPHEMISM (indirect, softening expression)
or LITERALLY (direct meaning)?

Classification:
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=20,
        )

        result = response.choices[0].message.content.strip()

        # Approximate cost tracking
        cost_per_call = 0.00025 + 0.00005  # ~$0.0003 per call
        total_cost += cost_per_call

        # Refusal detection
        refusal_phrases = ["cannot", "inappropriate", "i apologize", "i'm sorry"]
        if any(p in result.lower() for p in refusal_phrases):
            refusal_count += 1
            print(f"🚫 Refusal: '{result[:60]}...'")
            return "Refusal"

        result_lower = result.lower()

        if "euphemistic" in result_lower or "euphemism" in result_lower:
            return "Euphemistic"
        elif "literal" in result_lower:
            return "Literal"
        else:
            unexpected_count += 1
            print(f"⚠️ Unexpected response: '{result}' - defaulting to Literal")
            return "Unexpected"

    except Exception as e:
        print(f"❌ API Error: {e}")
        time.sleep(2)
        return "Error"


# ============================================================
# 🚀 FILE PROCESSING
# ============================================================

def process_file(file_config):
    """Process a single CSV file."""
    input_path = file_config["input"]
    output_path = file_config["output"]
    language = file_config["lang"]

    print("\n" + "=" * 70)
    print(f"🚀 Processing: {input_path}")
    print(f"   Language: {language}")
    print(f"   Model: {MODEL_NAME}")
    print("=" * 70)

    try:
        df = pd.read_csv(input_path)
        print(f"   ✓ Loaded {len(df)} examples")
    except FileNotFoundError:
        print("   ⚠️ File not found — skipping")
        return

    predictions = []

    file_refusals = 0
    file_unexpected = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Processing"):
        term = row[TERM_COL]
        sentence = clean_sentence(row[SENTENCE_COL], term)

        pred = classify_euphemism(term, sentence, language=language)
        predictions.append(pred)

        if pred == "Refusal":
            file_refusals += 1
        elif pred == "Unexpected":
            file_unexpected += 1

        time.sleep(0.1)

    df["gpt4_pred"] = predictions

    df["gpt4_binary"] = df["gpt4_pred"].apply(
        lambda x: 1 if x == "Euphemistic" else (0 if x == "Literal" else -1)
    )

    print("\n" + "─" * 60)
    print("   Statistics for this file:")
    print("─" * 60)
    print(f"   Total examples: {len(df)}")
    print(f"   Euphemistic: {(df['gpt4_pred'] == 'Euphemistic').sum()}")
    print(f"   Literal: {(df['gpt4_pred'] == 'Literal').sum()}")
    print(f"   Refusals: {file_refusals}")
    print(f"   Unexpected: {file_unexpected}")
    print(f"   Errors: {(df['gpt4_pred'] == 'Error').sum()}")

    if LABEL_COL in df.columns:
        valid_df = df[df["gpt4_pred"].isin(["Euphemistic", "Literal"])].copy()

        if len(valid_df) > 0:
            acc = (valid_df["gpt4_binary"] == valid_df[LABEL_COL]).mean()
            f1 = f1_score(valid_df[LABEL_COL], valid_df["gpt4_binary"])

            print(f"\n   Evaluation (on {len(valid_df)} valid predictions):")
            print(f"   Accuracy: {acc:.4f}")
            print(f"   F1 Score: {f1:.4f}")

            cm = confusion_matrix(valid_df[LABEL_COL], valid_df["gpt4_binary"])
            print("\n   Confusion Matrix:")
            print("                Predicted")
            print("              Lit    Euph")
            print(f"   Actual Lit [{cm[0,0]:4d}  {cm[0,1]:4d}]")
            print(f"          Euph [{cm[1,0]:4d}  {cm[1,1]:4d}]")
        else:
            print("   ⚠️ No valid predictions to evaluate")

    df.to_csv(output_path, index=False)
    print(f"\n   💾 Saved to {output_path}")


# ============================================================
# 🧪 MAIN
# ============================================================

def main():
    global total_cost

    print("=" * 70)
    print("GPT-4 Euphemism Classification Experiment")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    start_time = time.time()

    for config in files_to_process:
        process_file(config)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("🎉 EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total runtime: {elapsed / 60:.1f} minutes")
    print(f"Estimated cost: ${total_cost:.2f}")
    print(f"Total refusals: {refusal_count}")
    print(f"Total unexpected: {unexpected_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()

import os
import streamlit as st
import pandas as pd
import requests
import re
from fuzzywuzzy import process, fuzz
from time import sleep
from vertexai.generative_models import GenerativeModel
import vertexai

# ✅ Streamlit Config
st.set_page_config(layout="wide")
st.title("📦 AI-Powered HS Code Matcher (Live HTS API + Gemini Pro)")

# 📂 Upload Google Cloud Token and Project ID
st.sidebar.header("🔐 Google Cloud Configuration")
uploaded_token = st.sidebar.file_uploader("Upload your Google Cloud JSON key", type=["json"])
project_id = st.sidebar.text_input("Enter your Google Cloud Project ID")

if uploaded_token and project_id:
    token_path = "gcloud_key_uploaded.json"
    with open(token_path, "wb") as f:
        f.write(uploaded_token.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = token_path

    # 🧠 Gemini API function (dynamically uses user token/project)
    def query_gemini(prompt, project=project_id, location="us-central1"):
        vertexai.init(project=project, location=location)
        model = GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content(prompt)
        return response.text.strip()

    # 🔍 Get HS code and descriptions from live API (filtered for indent == 0)
    def get_hs_from_tariff_api(material):
        try:
            url = "https://hts.usitc.gov/reststop/search"
            response = requests.get(url, params={"keyword": material}, timeout=30)
            data = response.json()

            results = data.get("results", []) if isinstance(data, dict) else data if isinstance(data, list) else []

            final_results = []
            for item in results:
                if isinstance(item, dict) and item.get("indent") == "0" and item.get("htsno") and item.get("description"):
                    code = item["htsno"].strip()
                    desc = item["description"].strip()

                    if re.fullmatch(r"[0-9.]+", code) and len(code.replace(".", "")) < 10:
                        sub_response = requests.get(url, params={"keyword": code}, timeout=10)
                        sub_data = sub_response.json()
                        sub_results = sub_data.get("results", []) if isinstance(sub_data, dict) else sub_data

                        for sub_item in sub_results:
                            sub_code = sub_item.get("htsno", "").strip()
                            sub_desc = sub_item.get("description", "").strip()
                            if sub_code.startswith(code) and sub_code != code and len(sub_code.replace(".", "")) >= 8:
                                enriched_desc = f"{desc} - {sub_desc}"
                                final_results.append((sub_code, enriched_desc))
                    else:
                        final_results.append((code, desc))
            return final_results

        except Exception as e:
            st.warning(f"HTS API error for '{material}': {e}")
            return []

    # 🔍 Accurate fuzzy match with exact match priority
    def get_top_matches(query, choices, limit=3):
        flat_choices = [f"{code} - {desc}" for code, desc in choices]
        exact_matches = [desc for desc in flat_choices if query.lower() in desc.lower()]
        fuzzy_matches = process.extract(query, flat_choices, scorer=fuzz.token_set_ratio, limit=limit + 5)
        fuzzy_only = [match[0] for match in fuzzy_matches if match[0] not in exact_matches]
        top_combined = exact_matches + fuzzy_only
        top_combined = list(dict.fromkeys(top_combined))
        return [(desc, 100 if desc in exact_matches else process.extractOne(query, [desc], scorer=fuzz.token_set_ratio)[1]) for desc in top_combined[:limit]]

    # 🧠 Prompt for Gemini
    def ask_llama_for_best_code(material, candidates):
        prompt = f"""
You are a trade classification expert.

The user is trying to classify the material: "{material}"

Here are the 3 possible HS code matches:
{candidates}

Choose the best match from these codes. If you believe none of the given HS codes match **reasonably well**, use your best judgment to identify a more suitable HS code for the material from the HTS chapters. It should be 8 digits, in the format xxxx.xx.xx.

Do **not** mark "Not Found" unless the item is truly unrelated to any existing physical goods classification (e.g. pure software, services, subscriptions). When unsure, make an educated guess and explain the reasoning clearly.

You are allowed to suggest a better-fitting HS Code **even if it's not in the provided options**, if the description reasonably fits.

Mark "Not Found" **only** if the material cannot be classified under any HS code.

Please note that if the material description is a Non tangible item for e.g Services,Subscriptions etc, mark it "Non - Tangible item"
instead of "Not Found"

Your reply format must strictly be:

HS Code: <8-digit code in the format xxxx.xx.xx>
Reason: <why this code is correct>

Now for the material: "{material}"

Options:
{candidates}
        """
        return query_gemini(prompt)

    # 📤 Upload material file
    uploaded_file = st.file_uploader("📂 Upload CSV with a 'Material' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Material" not in df.columns:
            st.error("Uploaded CSV must have a column named 'Material'")
        else:
            st.subheader("🔍 Live API Lookup, Fuzzy Match, and AI Classification")
            results = []
            top3_table = []

            progress = st.progress(0, text="Starting AI matching...")
            status = st.empty()
            total = len(df)

            for idx, row in df.iterrows():
                material = row["Material"]
                status.text(f"Matching: {material} ({idx + 1}/{total})")

                api_results = get_hs_from_tariff_api(material)

                if not api_results:
                    results.append({
                        "Material": material,
                        "Matched HS Code": "Not found",
                        "Match Confidence (0-100)": "N/A",
                        "AI Explanation": "No API results"
                    })
                    continue

                matches = get_top_matches(material, api_results, limit=3)

                candidate_text = ""
                top_candidates = []
                for match_str, score in matches:
                    code, desc = match_str.split(" - ", 1)
                    candidate_text += f"{code} - {desc}\n"
                    top_candidates.append((code, desc, score))

                for code, desc, score in top_candidates:
                    top3_table.append({
                        "Material": material,
                        "HS Code": code,
                        "Description": desc,
                        "Score": score
                    })

                llama_response = ask_llama_for_best_code(material, candidate_text)

                matched_code = "Not found"
                explanation = llama_response
                match_score = "N/A"

                code_match = re.search(r"HS Code:\s*((?:[0-9]{4}(?:\.[0-9]{2}){2})|Non\s*-\s*Tangible\s*item|Not\s*Found)", llama_response)
                if code_match:
                    predicted_code = code_match.group(1)

                    matched_code = None
                    match_score = None

                    for code, desc, score in top_candidates:
                        if predicted_code == code:
                            matched_code = code
                            match_score = score
                            break

                    if matched_code is None:
                        matched_code = predicted_code
                        match_score = "N/A"

                results.append({
                    "Material": material,
                    "Matched HS Code": matched_code,
                    "Match Confidence (0-100)": match_score,
                    "AI Explanation": explanation
                })

                sleep(0.1)
                progress.progress((idx + 1) / total)

            # Preview Top Matches
            with st.expander("🔍 Preview: Top 3 Fuzzy Matches for Each Material"):
                st.dataframe(pd.DataFrame(top3_table))

            output_df = pd.DataFrame(results)
            st.success("✅ Matching complete!")
            st.dataframe(output_df)

            csv = output_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Result CSV", csv, "matched_hs_codes.csv", "text/csv")
else:
    st.info("⚠️ Please upload your Google Cloud JSON key and enter your Project ID in the sidebar to begin.")

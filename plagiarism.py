import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import uuid
import re
from collections import Counter

# Set up the app
st.set_page_config(page_title="Plagiarism Checker", layout="centered")
st.title("📚 Enhanced Assignment Plagiarism Checker")

# Initialize session state
if 'assignment_text' not in st.session_state:
    st.session_state.assignment_text = ""
if 'reference_texts' not in st.session_state:
    st.session_state.reference_texts = []

# --- Section 1: Upload Assignment ---
st.subheader("1️⃣ Upload or Paste Assignment")

file = st.file_uploader("Upload Assignment (.txt)", type=['txt'])
if file:
    st.session_state.assignment_text = file.read().decode("utf-8")

st.session_state.assignment_text = st.text_area(
    "Or paste assignment text here:",
    value=st.session_state.assignment_text,
    height=180
)

# --- Section 2: Upload References ---
st.subheader("2️⃣ Upload or Paste Reference Documents")

ref_files = st.file_uploader("Upload Reference Files", type=['txt'], accept_multiple_files=True)
for rf in ref_files:
    content = rf.read().decode("utf-8")
    st.session_state.reference_texts.append({
        'id': str(uuid.uuid4()),
        'name': rf.name,
        'text': content
    })

ref_input = st.text_area("Or paste reference text:", height=100)
if st.button("➕ Add Pasted Reference") and ref_input.strip():
    st.session_state.reference_texts.append({
        'id': str(uuid.uuid4()),
        'name': f"Manual Ref {len(st.session_state.reference_texts)+1}",
        'text': ref_input
    })
    st.success("Reference added successfully.")

# --- Display Reference List ---
if st.session_state.reference_texts:
    st.markdown("📚 *Current References:*")
    for ref in st.session_state.reference_texts:
        st.markdown(f"- {ref['name']} ({len(ref['text'].split())} words)")
    if st.button("🗑 Clear All References"):
        st.session_state.reference_texts = []

# --- Section 3: Run Plagiarism Check ---
st.subheader("3️⃣ Run Plagiarism Check")

threshold = st.slider("Set Similarity Highlight Threshold (%)", 0, 100, 70)

if st.button("🔍 Check Similarity"):
    if not st.session_state.assignment_text.strip():
        st.warning("Please provide assignment text.")
    elif not st.session_state.reference_texts:
        st.warning("Please add at least one reference.")
    else:
        all_docs = [st.session_state.assignment_text] + [r['text'] for r in st.session_state.reference_texts]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform(all_docs)

        results = []
        for i, ref in enumerate(st.session_state.reference_texts):
            sim = cosine_similarity(tfidf[0:1], tfidf[i+1:i+2])[0][0]
            results.append((ref['name'], sim))

        results.sort(key=lambda x: x[1], reverse=True)

        st.subheader("📊 Similarity Results")
        for name, score in results:
            percent = round(score * 100, 2)
            st.write(f"{name}** — Similarity: {percent}%")

        max_score = max([s for _, s in results], default=0)
        if max_score >= 0.75:
            st.error("⚠ High similarity detected. Possible plagiarism.")
        elif max_score >= 0.4:
            st.warning("⚠ Moderate similarity. Review suggested.")
        else:
            st.success("✅ Low similarity. Content likely original.")

        # --- Section 4: Stats & Readability ---
        st.subheader("📈 Assignment Analysis")

        def get_key_terms(text, top_n=10):
            words = re.findall(r'\b\w+\b', text.lower())
            stopwords = set([
                'the', 'is', 'and', 'in', 'to', 'of', 'that', 'with', 'for', 'as',
                'on', 'are', 'was', 'were', 'it', 'by', 'an', 'be', 'this', 'which',
                'or', 'from', 'a', 'at', 'but'
            ])
            filtered = [w for w in words if w not in stopwords]
            return [w for w, _ in Counter(filtered).most_common(top_n)]

        text = st.session_state.assignment_text
        word_count = len(text.split())
        sent_count = len(re.findall(r'[.!?]', text))
        read_score = textstat.flesch_kincaid_grade(text)
        key_terms = get_key_terms(text)

        st.markdown(f"- *Word Count:* {word_count}")
        st.markdown(f"- *Sentence Count:* {sent_count}")
        st.markdown(f"- *Flesch-Kincaid Grade Level:* {read_score}")
        st.markdown(f"- *Top Terms:* {', '.join(key_terms)}")

        # --- Section 5: Report Download ---
        if st.button("⬇ Download Report"):
            report = "📋 Plagiarism Report\\n\\n"
            for name, score in results:
                report += f"{name} - {round(score * 100, 2)}%\\n"
            report += f"\\nReadability Grade Level: {read_score}\\nKey Terms: {', '.join(key_terms)}"
            st.download_button("📄 Download", report, file_name="plagiarism_report.txt")
# BookLens - Book Analysis Web App

## Overview
BookLens is an NLP reading companion built as a course project. Upload a book (PDF, EPUB or TXT) and it detects people and places, picks out key details and an extractive summary, and shows the source sentences for those results. Processing runs on the host machine; no paid AI API is used.

The app is built to show its evidence. Characters, places, the summary and the key details are all tied back to sentences in the uploaded text. Results are statistical suggestions, not verified literary interpretation.

---

## Features
- Upload a searchable PDF, an unencrypted EPUB or a UTF-8 TXT file (English, up to 25 MB and 2 million characters). Nothing is silently truncated: a book that is too long is rejected with a message
- **Characters and places** ranked by mentions, with evidence sentences from the text
- **Summary**: original sentences selected across the book, in reading order
- **Key details**: source sentences ranked by vocabulary salience, plus ranked key phrases
- **NLP Analysis tab**: word and sentence counts, type-token ratio, frequent lemmas, part-of-speech counts and a plain-language description of the methods used
- **Local AI tab** (optional): a local Qwen2.5-7B model, run through Ollama, writes a section-based summary, suggests character roles and lists story locations by section. The summary is model-written text. Roles and locations are shown only with a cited sentence and a second-pass check, but they are model suggestions and can be wrong
- Download the analysis (and the Local AI result) as JSON
- Built-in sample text to try the app without a book

---

## How It Works

1. **Reading the book** (`analysis.py`, `epub_reader.py`)
   - PDF text is extracted page by page with pypdf. Encrypted PDFs are rejected
   - EPUB files are read offline from their reading order. Encrypted or DRM-protected files, oversized entries and unsafe XML declarations are rejected, and no files are extracted to disk
   - TXT files must be UTF-8
   - Project Gutenberg "START / END OF" license text is masked out when those markers are present
2. **Finding people and places** (`analysis.py`)
   - spaCy (`en_core_web_sm`) proposes named-entity candidates and proper nouns
   - Each candidate is then re-checked from context: locative words ("in", "at", "from"), use as the subject of a verb, and titles ("Mr", "Dr"). This decides whether a name is treated as a character, a place or neither (for example, date expressions are dropped)
   - Only spelling and case variants of a name are grouped. First names and surnames are not guessed to be the same person; possible aliases are flagged, not merged
3. **Summary and key details** (`analysis.py`)
   - TF-IDF (unigrams and bigrams, scikit-learn) scores vocabulary salience
   - The summary picks sentences across position bins of the book, then ranks by salience with a redundancy penalty. It returns original sentences, never generated text
4. **NLP statistics**: lemma frequencies, POS tag counts, type-token ratio and sentence counts
5. **Local AI (optional)** (`llm.py`, `jobs.py`)
   - The book is split into sections of about 9,000 characters. Each section is summarised, the section summaries are merged, and a final summary is written
   - For character roles, the model must cite one numbered sentence, a code check confirms the name appears in it, and a second model call must say the sentence states the role. Otherwise the role is left blank
   - For story locations, a section's place is kept only if the quoted sentence appears in that section, names the place, and passes the same second-pass check
   - A place-mention timeline shows where each place name first appears and how often it appears across ten equal slices of the book. It is a list of mentions, not a story travel timeline
   - The AI work runs as a background job so it continues if the page disconnects. Job results are kept in memory for 2 hours
   - The model talks only to a local Ollama server (`http://localhost:11434`)

---

## Tech Stack
- Python
- Streamlit
- spaCy (en_core_web_sm)
- scikit-learn
- pypdf
- Ollama with Qwen2.5-7B-Instruct (optional, Local AI tab only)

---

## How to Run the Project

1. Install Python 3 and create a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Download the English model: `python -m spacy download en_core_web_sm`
4. Start the app: `streamlit run app.py`
5. Open http://localhost:8501
6. Optional, for the Local AI tab: install Ollama, run `ollama pull qwen2.5:7b-instruct-q4_K_M` and keep the Ollama server running on port 11434. A full book can take many minutes on a laptop

---

## Project Structure

```
app.py            Streamlit interface and tabs
analysis.py       Book reading, entity detection, TF-IDF, extractive summary, statistics
epub_reader.py    Safe offline EPUB text reader
llm.py            Local Ollama client: summary, role checks, scene locations, place timeline
jobs.py           Background job runner for the Local AI tab
requirements.txt  Python dependencies
tests/            Unit tests for analysis, EPUB reading and the LLM layer
```

---

## Limitations
- English only. Scanned PDFs need OCR first
- Fictional names can be missed or misclassified; character ranking is by mentions, not narrative importance
- The extractive summary may omit plot links and keep unresolved pronouns, and it contains spoilers
- Local AI output is a model suggestion and can be wrong. It is slow on a full book
- Demo app: do not upload private documents

---

## Project Details

**Project Type:** NLP course project  
**Purpose:** Demonstrates NLP methods on full-length books: text cleaning, tokenization, frequency analysis, part-of-speech tagging, named-entity recognition, TF-IDF and extractive summarization.

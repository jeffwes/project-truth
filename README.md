# The Resonance Engine

**Version 3.1 (Final Integration)**  
**Project Name:** The Resonance Engine  
**AI Engine:** Google Gemini 3.0 Pro  
**Target Users:** Social Science Researchers & Media Analysts

## Executive Summary

The Resonance Engine is a local-first desktop application designed to deconstruct the persuasive architecture of modern media. Unlike standard summarizers that tell you *what* an article says, this tool tells you *how it works*.

It operates as a "second brain" that ingests content from active browser tabs, YouTube videos, or raw text, then uses Gemini 3.0 to:
1. Separate verifiable reality from shared myths ("inter-subjective" reality)
2. Map content to specific psychological triggers (Moral Foundations)
3. Visualize engine data to predict which social "tribes" the content will resonate with

## Theoretical Frameworks

### A. The Reality Taxonomy (Yuval Noah Harari)

Based on *Sapiens*, this framework classifies every assertion into:

- **Objective Reality**: Phenomena that exist independently of human consciousness
- **Subjective Reality**: Phenomena that exist only in individual consciousness  
- **Intersubjective Reality**: Phenomena sustained by shared beliefs

### B. Moral Foundations Theory (Jonathan Haidt)

Six psychological triggers: Care, Fairness, Loyalty, Authority, Sanctity, Liberty

### C. Tribal Resonance Prediction

Predicts which social/political tribes will find content compelling.

## Installation

1. Ensure `.env` contains: `GEMINI_API_KEY=your_key`
2. Install: `pip install -r requirements.txt`
3. Run: `shiny run app.py`
4. Open: http://127.0.0.1:8000

## Usage

1. Select input mode (text/URL/YouTube)
2. Enter content
3. Click "Analyze Content"
4. Explore results in tabs
5. Export as JSON or CSV

## Architecture

- `src/gemini_client.py` - API client
- `src/reality_taxonomy.py` - Harari analysis
- `src/moral_foundations.py` - MFT analysis
- `src/tribal_resonance.py` - Audience prediction
- `app.py` - Shiny UI

---

Version 3.1 | November 25, 2025

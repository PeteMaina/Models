import streamlit as st
import PyPDF2
import docx
import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization import keywords
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import networkx as nx
from heapq import nlargest
import textstat
import io

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize spaCy
nlp = spacy.load('en_core_web_sm')

class SmartDocumentSummarizer:
    def __init__(self):
        self.text = ""

    def read_pdf(self, file):
        """Extract text from PDF files"""
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def read_docx(self, file):
        """Extract text from DOCX files"""
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    def read_txt(self, file):
        """Read text files"""
        text = file.getvalue().decode("utf-8")
        return text
    
    def preprocess_text(self, text):
        """Preprocess text for summarization"""
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'\n+', ' ', text)  # Remove newlines
        return text
    
    def get_sentences(self, text):
        """Split text into sentences"""
        return sent_tokenize(text)
    
    def extract_keywords(self, text, num_keywords=10):
        """Extract key terms from the text"""
        doc = nlp(text)
        words = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN', 'ADJ']]
        word_freq = Counter(words)
        return word_freq.most_common(num_keywords)
    
    def summarize_text_ratio(self, text, ratio=0.2):
        """Summarize text based on a ratio of the original length"""
        # Preprocess
        sentences = self.get_sentences(text)
        if not sentences:
            return "The document appears to be empty or couldn't be properly parsed."
        
        # Calculate number of sentences for summary
        num_sentences = max(1, int(len(sentences) * ratio))
        
        # Use TextRank algorithm for summarization
        return self.text_rank_summarize(text, num_sentences)
    
    def summarize_pareto(self, text):
        """Apply Pareto principle (80/20 rule) for summarization"""
        # Summarize using 20% of the content
        return self.summarize_text_ratio(text, 0.2)
    
    def summarize_custom(self, text, summary_size):
        """Summarize text with custom size"""
        sentences = self.get_sentences(text)
        if not sentences:
            return "The document appears to be empty or couldn't be properly parsed."
        
        # Calculate number of sentences for summary (ensure at least 1)
        num_sentences = max(1, min(summary_size, len(sentences)))
        
        # Use TextRank for summarization
        return self.text_rank_summarize(text, num_sentences)
    
    def text_rank_summarize(self, text, num_sentences):
        """Implement TextRank algorithm for extractive summarization"""
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # If there are fewer sentences than requested, return them all
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Create similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(sentences[i], sentences[j])
        
        # Convert similarity matrix to graph
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Get top sentences
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        
        # Select top sentences but preserve original order
        selected_indices = [item[1] for item in ranked_sentences[:num_sentences]]
        selected_indices.sort()
        summary = [sentences[i] for i in selected_indices]
        
        return " ".join(summary)
    
    def sentence_similarity(self, sent1, sent2):
        """Calculate similarity between two sentences"""
        doc1 = nlp(sent1)
        doc2 = nlp(sent2)
        
        # If either sentence is parsed as empty, return 0
        if not doc1 or not doc2:
            return 0
            
        # Use spaCy's vector representation for similarity
        if doc1.vector_norm and doc2.vector_norm:  # Check if vectors exist
            return doc1.similarity(doc2)
        return 0
    
    def analyze_document(self, text):
        """Analyze document for metadata and statistics"""
        # Count tokens
        tokens = word_tokenize(text)
        words = [word.lower() for word in tokens if word.isalpha()]
        
        # Calculate readability scores
        readability = {
            "Flesch Reading Ease": textstat.flesch_reading_ease(text),
            "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
            "SMOG Index": textstat.smog_index(text),
            "Coleman-Liau Index": textstat.coleman_liau_index(text),
            "Automated Readability Index": textstat.automated_readability_index(text)
        }
        
        # Get word frequency
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        word_freq = Counter(filtered_words)
        
        # Calculate sentence stats
        sentences = sent_tokenize(text)
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        
        stats = {
            "Total Words": len(words),
            "Unique Words": len(set(words)),
            "Total Sentences": len(sentences),
            "Average Words per Sentence": sum(sentence_lengths) / max(1, len(sentence_lengths)),
            "Readability Scores": readability,
            "Top Words": word_freq.most_common(10)
        }
        
        return stats
    
    def generate_topic_clusters(self, text, num_topics=5):
        """Generate topic clusters from document"""
        # Preprocessing
        sentences = sent_tokenize(text)
        
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.7,
            min_df=3
        )
        
        # Create vectors
        if len(sentences) < 5:  # Need sufficient sentences for meaningful clustering
            return [{"topic": "Main Topic", "keywords": [w for w, _ in self.extract_keywords(text, 5)]}]
            
        try:
            X = vectorizer.fit_transform(sentences)
            
            # Simple clustering based on similarity
            topic_clusters = []
            feature_names = vectorizer.get_feature_names_out()
            
            for i in range(min(num_topics, len(sentences) // 2)):
                # Get the most significant terms for this "topic"
                if i < X.shape[0]:
                    tfidf_sorting = np.argsort(X.toarray()[i])[::-1]
                    top_n = 5  # Top N keywords for each topic
                    top_features = [feature_names[j] for j in tfidf_sorting[:top_n]]
                    
                    topic_clusters.append({
                        "topic": f"Topic {i+1}",
                        "keywords": top_features
                    })
            
            return topic_clusters
        except:
            # Fallback if clustering fails
            return [{"topic": "Main Topic", "keywords": [w for w, _ in self.extract_keywords(text, 5)]}]

    def hierarchical_summarize(self, text):
        """Create a hierarchical summary with different levels of detail"""
        # Create three levels of summary
        high_level = self.summarize_text_ratio(text, 0.1)  # Very brief
        mid_level = self.summarize_text_ratio(text, 0.2)   # Medium detail
        detailed = self.summarize_text_ratio(text, 0.3)    # More comprehensive
        
        return {
            "high_level": high_level,
            "mid_level": mid_level,
            "detailed": detailed
        }

    def get_key_insights(self, text, num_insights=5):
        """Extract key insights from the document"""
        # Preprocess and split into sentences
        sentences = self.get_sentences(text)
        
        # Use TextRank to find important sentences
        doc = nlp(text)
        sentence_scores = {}
        
        for sent in doc.sents:
            # Skip very short sentences
            if len(sent.text.split()) < 3:
                continue
                
            for token in sent:
                if not token.is_stop and not token.is_punct and token.has_vector:
                    if sent in sentence_scores:
                        sentence_scores[sent] += token.vector_norm
                    else:
                        sentence_scores[sent] = token.vector_norm
        
        # Get sentences with highest scores
        try:
            key_sentences = nlargest(num_insights, sentence_scores, key=sentence_scores.get)
            insights = [sent.text.strip() for sent in key_sentences]
        except:
            # Fallback if the above fails
            insights = sentences[:num_insights] if len(sentences) >= num_insights else sentences
            
        return insights


# Streamlit UI
def main():
    st.set_page_config(page_title="Smart Document Summarizer", layout="wide")
    
    st.title("📚 Smart Document Summarizer")
    st.markdown("""
    Upload any document (PDF, DOCX, TXT) to get smart summaries using various techniques.
    This tool uses advanced NLP algorithms to extract key information from your documents.
    """)
    
    # Initialize session state for storing the document text
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'file_name' not in st.session_state:
        st.session_state.file_name = ""
    if 'document_stats' not in st.session_state:
        st.session_state.document_stats = None
    if 'show_summary' not in st.session_state:
        st.session_state.show_summary = False
        
    # Initialize summarizer
    summarizer = SmartDocumentSummarizer()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your document", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Extract text based on file type
            if uploaded_file.name.endswith('.pdf'):
                text = summarizer.read_pdf(uploaded_file)
            elif uploaded_file.name.endswith('.docx'):
                text = summarizer.read_docx(uploaded_file)
            else:  # Assume txt
                text = summarizer.read_txt(uploaded_file)
            
            # Clean the text
            text = summarizer.preprocess_text(text)
            
            # Store in session state
            st.session_state.document_text = text
            st.session_state.file_name = uploaded_file.name
            
            # Analyze document
            st.session_state.document_stats = summarizer.analyze_document(text)
            st.session_state.show_summary = True
    
    # If text is available, show the summary options
    if st.session_state.show_summary and st.session_state.document_text:
        st.success(f"Successfully processed: {st.session_state.file_name}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Document Analysis", "Advanced Options", "Full Text"])
        
        with tab1:
            st.header("Document Summary")
            
            # Summary method selection
            summary_method = st.selectbox(
                "Choose a summarization method:",
                ["Pareto Principle (80/20 Rule)", "Custom Length", "Hierarchical Summary"]
            )
            
            if summary_method == "Pareto Principle (80/20 Rule)":
                st.info("The Pareto Principle (80/20 rule) suggests that 80% of the information can be found in 20% of the content.")
                
                if st.button("Generate Pareto Summary"):
                    with st.spinner("Generating summary using Pareto principle..."):
                        summary = summarizer.summarize_pareto(st.session_state.document_text)
                        st.markdown("### Summary (20% of original)")
                        st.write(summary)
                        
                        # Show extraction ratio
                        original_len = len(word_tokenize(st.session_state.document_text))
                        summary_len = len(word_tokenize(summary))
                        st.caption(f"Summary: {summary_len} words ({(summary_len/original_len*100):.1f}% of original {original_len} words)")
            
            elif summary_method == "Custom Length":
                # Get sentence count for reference
                sentence_count = len(sent_tokenize(st.session_state.document_text))
                st.info(f"The document has approximately {sentence_count} sentences.")
                
                # Custom summary length
                summary_size = st.slider("Select number of sentences for summary:", 
                                         min_value=1, 
                                         max_value=max(50, sentence_count),
                                         value=min(10, sentence_count))
                
                if st.button("Generate Custom Summary"):
                    with st.spinner("Generating custom summary..."):
                        summary = summarizer.summarize_custom(st.session_state.document_text, summary_size)
                        st.markdown(f"### Summary ({summary_size} sentences)")
                        st.write(summary)
                        
                        # Show extraction ratio
                        original_len = len(word_tokenize(st.session_state.document_text))
                        summary_len = len(word_tokenize(summary))
                        st.caption(f"Summary: {summary_len} words ({(summary_len/original_len*100):.1f}% of original {original_len} words)")
            
            else:  # Hierarchical Summary
                st.info("Hierarchical summary provides different levels of detail.")
                
                if st.button("Generate Hierarchical Summary"):
                    with st.spinner("Generating hierarchical summary..."):
                        hierarchical = summarizer.hierarchical_summarize(st.session_state.document_text)
                        
                        st.markdown("### Executive Summary (10%)")
                        st.write(hierarchical["high_level"])
                        
                        st.markdown("### Standard Summary (20%)")
                        st.write(hierarchical["mid_level"])
                        
                        st.markdown("### Detailed Summary (30%)")
                        st.write(hierarchical["detailed"])
            
            # Key insights
            st.markdown("### Key Insights")
            num_insights = st.slider("Number of key insights to extract:", 3, 10, 5)
            
            if st.button("Extract Key Insights"):
                with st.spinner("Extracting key insights..."):
                    insights = summarizer.get_key_insights(st.session_state.document_text, num_insights)
                    for i, insight in enumerate(insights, 1):
                        st.markdown(f"**{i}.** {insight}")
        
        with tab2:
            st.header("Document Analysis")
            
            if st.session_state.document_stats:
                stats = st.session_state.document_stats
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Document Statistics")
                    st.write(f"**Total Words:** {stats['Total Words']:,}")
                    st.write(f"**Unique Words:** {stats['Unique Words']:,}")
                    st.write(f"**Total Sentences:** {stats['Total Sentences']:,}")
                    st.write(f"**Average Words per Sentence:** {stats['Average Words per Sentence']:.2f}")
                
                with col2:
                    st.subheader("Readability Scores")
                    readability = stats['Readability Scores']
                    
                    # Interpret Flesch Reading Ease score
                    flesch = readability['Flesch Reading Ease']
                    interpretation = ""
                    if flesch >= 90:
                        interpretation = "Very Easy (5th grade)"
                    elif flesch >= 80:
                        interpretation = "Easy (6th grade)"
                    elif flesch >= 70:
                        interpretation = "Fairly Easy (7th grade)"
                    elif flesch >= 60:
                        interpretation = "Standard (8-9th grade)"
                    elif flesch >= 50:
                        interpretation = "Fairly Difficult (10-12th grade)"
                    elif flesch >= 30:
                        interpretation = "Difficult (College level)"
                    else:
                        interpretation = "Very Difficult (College graduate)"
                        
                    st.write(f"**Flesch Reading Ease:** {flesch:.1f} ({interpretation})")
                    st.write(f"**Flesch-Kincaid Grade:** {readability['Flesch-Kincaid Grade']:.1f}")
                    st.write(f"**SMOG Index:** {readability['SMOG Index']:.1f}")
                    st.write(f"**Coleman-Liau Index:** {readability['Coleman-Liau Index']:.1f}")
                    st.write(f"**Automated Readability Index:** {readability['Automated Readability Index']:.1f}")
                
                # Visualize top words
                st.subheader("Frequently Used Words")
                top_words = stats['Top Words']
                
                # Convert to dataframe for visualization
                word_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(word_df['Word'], word_df['Count'], color='skyblue')
                ax.set_xlabel('Frequency')
                ax.set_title('Top Words by Frequency')
                
                # Add labels to bars
                for bar in bars:
                    width = bar.get_width()
                    label_position = width + 0.5
                    ax.text(label_position, bar.get_y() + bar.get_height()/2, f'{width:.0f}', 
                            ha='left', va='center')
                
                st.pyplot(fig)
                
                # Topic analysis
                st.subheader("Topic Analysis")
                topics = summarizer.generate_topic_clusters(st.session_state.document_text)
                
                for topic in topics:
                    st.write(f"**{topic['topic']}:** {', '.join(topic['keywords'])}")
                
        with tab3:
            st.header("Advanced Options")
            
            # Keyword extraction
            st.subheader("Keyword Extraction")
            keyword_count = st.slider("Number of keywords to extract:", 5, 30, 15)
            
            if st.button("Extract Keywords"):
                with st.spinner("Extracting keywords..."):
                    keywords = summarizer.extract_keywords(st.session_state.document_text, keyword_count)
                    
                    # Create columns for keyword display
                    cols = st.columns(3)
                    for i, (word, count) in enumerate(keywords):
                        col_idx = i % 3
                        cols[col_idx].metric(f"Keyword {i+1}", word, f"Count: {count}")
            
            # Custom ratio summarization
            st.subheader("Custom Ratio Summarization")
            ratio = st.slider("Select summary ratio (% of original text):", 5, 50, 20)
            
            if st.button("Generate Ratio-Based Summary"):
                with st.spinner(f"Generating summary ({ratio}% of original)..."):
                    summary = summarizer.summarize_text_ratio(st.session_state.document_text, ratio/100)
                    st.markdown(f"### {ratio}% Summary")
                    st.write(summary)
                    
                    # Show extraction ratio
                    original_len = len(word_tokenize(st.session_state.document_text))
                    summary_len = len(word_tokenize(summary))
                    st.caption(f"Summary: {summary_len} words ({(summary_len/original_len*100):.1f}% of original {original_len} words)")
        
        with tab4:
            st.header("Full Document Text")
            st.info("This shows the raw extracted text from your document.")
            
            # Add a search box for the full text
            search_term = st.text_input("Search in document:")
            
            # Show document with search highlighting if needed
            if search_term:
                # Highlight occurrences of search term
                highlighted_text = st.session_state.document_text.replace(
                    search_term, 
                    f"**{search_term}**"
                )
                st.markdown(highlighted_text)
                
                # Count occurrences
                occurrences = st.session_state.document_text.lower().count(search_term.lower())
                st.caption(f"Found {occurrences} occurrences of '{search_term}'")
            else:
                st.text_area("Document Content", st.session_state.document_text, height=400)
            
            # Download the extracted text
            text_data = io.StringIO(st.session_state.document_text)
            st.download_button(
                label="Download Extracted Text",
                data=text_data,
                file_name=f"{st.session_state.file_name.split('.')[0]}_extracted.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()

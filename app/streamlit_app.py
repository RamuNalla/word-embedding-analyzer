"""
Streamlit application for word embedding visualization and analysis.
"""

import streamlit as st
import yaml
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.torchtext_glove_embeddings import TorchTextGloVeEmbedding
from src.embeddings.torchtext_word2vec_embeddings import TorchTextFastTextEmbedding
from src.tasks.similarity import calculate_similarity, get_most_similar, similarity_matrix
from src.tasks.analogy import solve_analogy, simple_analogy, compare_analogies
from src.visualization.visualizer import EmbeddingVisualizer


# Page configuration
st.set_page_config(
    page_title="Word Embedding Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_config():
    """Load configuration file."""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


@st.cache_resource
def load_embeddings():
    """Load embedding models."""
    embeddings = {}
    config = load_config()
    
    # Try TorchText GloVe (automatic download)
    if config.get('embeddings', {}).get('torchtext_glove', {}).get('enabled', True):
        with st.spinner("Loading TorchText GloVe model (will download if needed)..."):
            try:
                glove_config = config.get('embeddings', {}).get('torchtext_glove', {})
                name = glove_config.get('name', '6B')
                dim = glove_config.get('dim', 100)
                cache = glove_config.get('cache', '.vector_cache')
                
                embeddings[f'GloVe-{name}-{dim}d'] = TorchTextGloVeEmbedding(
                    name=name, dim=dim, cache=cache
                )
                st.success(f"✅ TorchText GloVe {name} {dim}d loaded")
            except Exception as e:
                st.warning(f"⚠️ Could not load TorchText GloVe: {e}")
    
    # Try TorchText FastText (automatic download)
    if config.get('embeddings', {}).get('torchtext_fasttext', {}).get('enabled', False):
        with st.spinner("Loading TorchText FastText model (will download if needed)..."):
            try:
                fasttext_config = config.get('embeddings', {}).get('torchtext_fasttext', {})
                language = fasttext_config.get('language', 'en')
                cache = fasttext_config.get('cache', '.vector_cache')
                
                embeddings[f'FastText-{language}'] = TorchTextFastTextEmbedding(
                    language=language, cache=cache
                )
                st.success(f"✅ TorchText FastText {language} loaded")
            except Exception as e:
                st.warning(f"⚠️ Could not load TorchText FastText: {e}")
    
    return embeddings





def main():
    """Main application function."""
    
    # Title and description
    st.title("📊 Word Embedding Analyzer")
    st.markdown("Compare Word2Vec and GloVe embeddings for similarity and analogy tasks")
    
    # Load embeddings
    embeddings = load_embeddings()
    
    if not embeddings:
        st.error("❌ No embedding models could be loaded. Please check your model files.")
        st.info("Run `python scripts/download_models.py` to download pre-trained models.")
        return
    
    # Sidebar - Model selection
    st.sidebar.header("⚙️ Settings")
    
    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=list(embeddings.keys()),
        default=list(embeddings.keys())
    )
    
    if not selected_models:
        st.warning("Please select at least one model from the sidebar.")
        return
    
    # Main content - Task selection
    task = st.sidebar.radio(
        "Select Task",
        ["Word Similarity", "Most Similar Words", "Word Analogy", "Visualization"]
    )
    
    st.sidebar.markdown("---")
    
    # Display model info
    with st.sidebar.expander("📋 Model Information"):
        for name, emb in embeddings.items():
            if name in selected_models:
                st.write(f"**{name}**")
                st.write(f"- Vocabulary: {emb.get_vocabulary_size():,} words")
                st.write(f"- Vector Size: {emb.vector_size}")
    
    # Task implementations
    if task == "Word Similarity":
        word_similarity_task(embeddings, selected_models)
    
    elif task == "Most Similar Words":
        most_similar_task(embeddings, selected_models)
    
    elif task == "Word Analogy":
        analogy_task(embeddings, selected_models)
    
    elif task == "Visualization":
        visualization_task(embeddings, selected_models)


def word_similarity_task(embeddings, selected_models):
    """Word similarity task interface."""
    st.header("🔍 Word Similarity")
    st.markdown("Calculate cosine similarity between two words.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        word1 = st.text_input("First Word", value="king", key="sim_word1")
    
    with col2:
        word2 = st.text_input("Second Word", value="queen", key="sim_word2")
    
    if st.button("Calculate Similarity", type="primary"):
        if not word1 or not word2:
            st.warning("Please enter both words.")
            return
        
        results = []
        
        for model_name in selected_models:
            embedding = embeddings[model_name]
            
            if not embedding.contains(word1):
                st.error(f"'{word1}' not in {model_name} vocabulary")
                continue
            
            if not embedding.contains(word2):
                st.error(f"'{word2}' not in {model_name} vocabulary")
                continue
            
            similarity = calculate_similarity(embedding, word1, word2)
            results.append({
                'Model': model_name,
                'Similarity': f"{similarity:.4f}",
                'Score': similarity
            })
        
        if results:
            df = pd.DataFrame(results)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Results")
                st.dataframe(df[['Model', 'Similarity']], use_container_width=True)
            
            with col2:
                st.subheader("Comparison")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(df['Model'], df['Score'], color=['#1f77b4', '#ff7f0e'])
                ax.set_xlabel('Similarity Score')
                ax.set_xlim(0, 1)
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
            
            # Winner
            best_model = df.loc[df['Score'].idxmax(), 'Model']
            best_score = df.loc[df['Score'].idxmax(), 'Score']
            st.success(f"🏆 Highest similarity: **{best_model}** ({best_score:.4f})")


def most_similar_task(embeddings, selected_models):
    """Most similar words task interface."""
    st.header("🎯 Most Similar Words")
    st.markdown("Find words most similar to a given word.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        word = st.text_input("Enter Word", value="computer", key="similar_word")
    
    with col2:
        topn = st.slider("Number of Results", 5, 20, 10)
    
    if st.button("Find Similar Words", type="primary"):
        if not word:
            st.warning("Please enter a word.")
            return
        
        tabs = st.tabs(selected_models)
        
        for tab, model_name in zip(tabs, selected_models):
            with tab:
                embedding = embeddings[model_name]
                
                if not embedding.contains(word):
                    st.error(f"'{word}' not in {model_name} vocabulary")
                    continue
                
                similar = get_most_similar(embedding, word, topn=topn)
                
                # Create DataFrame
                df = pd.DataFrame(similar, columns=['Word', 'Similarity'])
                df['Rank'] = range(1, len(df) + 1)
                df = df[['Rank', 'Word', 'Similarity']]
                
                # Display table
                st.dataframe(df, use_container_width=True)
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(df['Word'][::-1], df['Similarity'][::-1])
                ax.set_xlabel('Similarity Score')
                ax.set_title(f'Top {topn} Similar Words to "{word}" - {model_name}')
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)


def analogy_task(embeddings, selected_models):
    """Word analogy task interface."""
    st.header("🧩 Word Analogy")
    st.markdown("Solve word analogies using vector arithmetic.")
    
    # Choose analogy mode
    mode = st.radio("Analogy Mode", ["Simple (A:B :: C:?)", "Custom"], horizontal=True)
    
    if mode == "Simple (A:B :: C:?)":
        st.markdown("**Format:** A is to B as C is to ?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            a = st.text_input("A", value="king", key="a")
        with col2:
            b = st.text_input("B", value="queen", key="b")
        with col3:
            c = st.text_input("C", value="man", key="c")
        
        positive = [b, c]
        negative = [a]
        analogy_str = f"{a} : {b} :: {c} : ?"
    
    else:
        st.markdown("**Vector arithmetic:** positive - negative")
        
        col1, col2 = st.columns(2)
        
        with col1:
            positive_input = st.text_input("Positive words (comma-separated)", 
                                          value="king,woman", key="pos")
            positive = [w.strip() for w in positive_input.split(',') if w.strip()]
        
        with col2:
            negative_input = st.text_input("Negative words (comma-separated)", 
                                          value="man", key="neg")
            negative = [w.strip() for w in negative_input.split(',') if w.strip()]
        
        analogy_str = f"{' + '.join(positive)} - {' - '.join(negative)} = ?"
    
    topn = st.slider("Number of Results", 1, 10, 5, key="analogy_topn")
    
    if st.button("Solve Analogy", type="primary"):
        if not positive or not negative:
            st.warning("Please provide both positive and negative words.")
            return
        
        st.subheader(f"Analogy: {analogy_str}")
        
        tabs = st.tabs(selected_models)
        
        for tab, model_name in zip(tabs, selected_models):
            with tab:
                embedding = embeddings[model_name]
                
                # Check vocabulary
                all_words = positive + negative
                missing = [w for w in all_words if not embedding.contains(w)]
                
                if missing:
                    st.error(f"Words not in vocabulary: {', '.join(missing)}")
                    continue
                
                results = solve_analogy(embedding, positive, negative, topn=topn)
                
                if results:
                    # Display results
                    df = pd.DataFrame(results, columns=['Word', 'Score'])
                    df['Rank'] = range(1, len(df) + 1)
                    df = df[['Rank', 'Word', 'Score']]
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Highlight top result
                    st.success(f"✨ Top answer: **{results[0][0]}** (score: {results[0][1]:.4f})")
                else:
                    st.warning("No results found.")


def visualization_task(embeddings, selected_models):
    """Visualization task interface."""
    st.header("📈 Embedding Visualization")
    
    if len(selected_models) == 0:
        st.warning("Please select at least one model.")
        return
    
    model_name = st.selectbox("Select Model", selected_models)
    embedding = embeddings[model_name]
    visualizer = EmbeddingVisualizer(embedding)
    
    viz_type = st.radio("Visualization Type", 
                       ["Word Clusters", "Similarity Heatmap", "Word Neighbors"],
                       horizontal=True)
    
    if viz_type == "Word Clusters":
        st.subheader("Word Cluster Visualization")
        
        words_input = st.text_area(
            "Enter words (one per line or comma-separated)",
            value="king,queen,man,woman,prince,princess,boy,girl",
            height=100
        )
        
        # Parse words
        if ',' in words_input:
            words = [w.strip() for w in words_input.split(',') if w.strip()]
        else:
            words = [w.strip() for w in words_input.split('\n') if w.strip()]
        
        method = st.selectbox("Reduction Method", ["tsne", "pca"])
        
        if st.button("Generate Visualization", type="primary"):
            if len(words) < 2:
                st.warning("Please enter at least 2 words.")
                return
            
            with st.spinner("Generating visualization..."):
                try:
                    fig = visualizer.plot_words_2d(words, method=method, figsize=(12, 8))
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    elif viz_type == "Similarity Heatmap":
        st.subheader("Similarity Heatmap")
        
        words_input = st.text_area(
            "Enter words (one per line or comma-separated)",
            value="king,queen,man,woman,boy,girl",
            height=100
        )
        
        # Parse words
        if ',' in words_input:
            words = [w.strip() for w in words_input.split(',') if w.strip()]
        else:
            words = [w.strip() for w in words_input.split('\n') if w.strip()]
        
        if st.button("Generate Heatmap", type="primary"):
            if len(words) < 2:
                st.warning("Please enter at least 2 words.")
                return
            
            with st.spinner("Generating heatmap..."):
                try:
                    fig = visualizer.plot_similarity_heatmap(words, figsize=(10, 8))
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    elif viz_type == "Word Neighbors":
        st.subheader("Word Neighbors Visualization")
        
        word = st.text_input("Enter Word", value="computer")
        topn = st.slider("Number of Neighbors", 5, 20, 10)
        method = st.selectbox("Reduction Method", ["tsne", "pca"])
        
        if st.button("Visualize Neighbors", type="primary"):
            if not word:
                st.warning("Please enter a word.")
                return
            
            with st.spinner("Generating visualization..."):
                try:
                    fig = visualizer.plot_word_neighbors(word, topn=topn, method=method, figsize=(12, 8))
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
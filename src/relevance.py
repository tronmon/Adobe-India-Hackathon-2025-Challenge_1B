from sentence_transformers import SentenceTransformer, util
import nltk
import torch

class RelevanceAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initializes the analyzer and loads the sentence transformer model."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')

    def rank_sections(self, sections, relevance_profile):
        """Ranks sections based on semantic similarity to the relevance profile."""
        if not sections:
            return []

        profile_embedding = self.model.encode(relevance_profile, convert_to_tensor=True)
        
        section_texts = [sec['section_text'] for sec in sections]
        section_embeddings = self.model.encode(section_texts, convert_to_tensor=True)

        # Calculate cosine similarities
        cosine_scores = util.cos_sim(profile_embedding, section_embeddings)[0]

        for i, section in enumerate(sections):
            section['relevance_score'] = cosine_scores[i].item()

        # Sort sections by relevance score in descending order
        ranked_sections = sorted(sections, key=lambda x: x['relevance_score'], reverse=True)
        
        # Add importance rank
        for i, section in enumerate(ranked_sections):
            section['importance_rank'] = i + 1
            
        return ranked_sections

    def analyze_subsections(self, ranked_sections, relevance_profile, top_n_sections=5, top_k_sentences=5):
        """Performs extractive summarization on the most relevant sections."""
        analysis_results = []
        profile_embedding = self.model.encode(relevance_profile, convert_to_tensor=True)

        # Process only the top N most relevant sections
        for section in ranked_sections[:top_n_sections]:
            text = section['section_text']
            sentences = nltk.sent_tokenize(text)

            if not sentences:
                continue

            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            
            # Score each sentence against the profile
            cosine_scores = util.cos_sim(profile_embedding, sentence_embeddings)[0]

            # Get top K sentences
            top_sentence_indices = torch.topk(cosine_scores, k=min(top_k_sentences, len(sentences)), sorted=True).indices

            # Sort indices to maintain original sentence order
            sorted_indices = sorted([idx.item() for idx in top_sentence_indices])
            
            refined_text = " ".join([sentences[i] for i in sorted_indices])
            
            analysis_results.append({
                "document": section['document'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })

        return analysis_results



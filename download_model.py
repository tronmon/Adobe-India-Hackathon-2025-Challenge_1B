from sentence_transformers import SentenceTransformer
import nltk

def main():
    """
    Downloads and caches the required models for offline use.
    This script is intended to be run during the Docker build process.
    """
    print("Downloading sentence-transformer model: all-MiniLM-L6-v2")
    SentenceTransformer('all-MiniLM-L6-v2')
    print("Model downloaded and cached.")

    print("Downloading NLTK data: punkt")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    print("NLTK data downloaded and cached.")

if __name__ == "__main__":
    main()


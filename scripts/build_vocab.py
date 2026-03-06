import os
import json
from collections import Counter
import re

def extract_vocab():
    print("Extracting top 10,000 words from TinyStories...")
    data_path = 'data/TinyStories-train.txt'
    out_path = 'data/core_concepts.txt'
    
    with open(data_path, 'r', encoding='utf-8') as f:
        # Read a chunk to get a good representative vocab
        text = f.read(50000000)  # 50MB
        
    words = re.findall(r'\b[a-zA-Z]{3,12}\b', text.lower())
    counts = Counter(words)
    
    # Take top 10000 most common words, skip top 50 (stopwords)
    most_common = counts.most_common(10050)[50:]
    
    # Add our specific test words to ensure they are present for Step A
    test_words = [
        "king", "queen", "duke", "prince", "emperor", "baron", "monarch",
        "dog", "cat", "wolf", "fox", "puppy", "hound", "pet",
        "house", "castle", "mansion", "cabin", "hut", "apartment", "building",
        "water", "fire", "ocean", "river", "rain", "drink", "liquid",
        "doctor", "hospital", "sun", "moon", "happy", "sad", "mother", "father"
    ]
    
    final_words = set(w[0].capitalize() for w in most_common)
    for w in test_words:
        final_words.add(w.capitalize())
        
    with open(out_path, 'w', encoding='utf-8') as f:
        for w in sorted(list(final_words)):
            f.write(w + '\n')
            
    print(f"Saved {len(final_words)} concepts to {out_path}")

if __name__ == '__main__':
    extract_vocab()

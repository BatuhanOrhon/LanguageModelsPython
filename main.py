import nltk
import re
from collections import defaultdict, Counter
from itertools import tee

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

def read_file(file_path):
    """Reads the content of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def count_sentences(text):
    """Counts the number of sentences ending with '.', '!', or '?'."""
    return len(re.findall(r'[.!?]', text))

def tokenize_and_tag_sentences(text):
    """Tokenizes the text, adds <s> and </s> to mark sentence boundaries."""
    tokens = nltk.word_tokenize(text, language='turkish')
    tokens = [token.lower() for token in tokens]

    processed_tokens = []
    current_sentence = []
    
    for token in tokens:
        if token in '.!?':
            if current_sentence:
                processed_tokens.extend(['<s>'] + current_sentence + ['</s>'])
                current_sentence = []
        else:
            current_sentence.append(token)

    if current_sentence:  # Handle the last sentence if not ended with punctuation
        processed_tokens.extend(['<s>'] + current_sentence + ['</s>'])

    return processed_tokens

def compute_unigrams(tokens):
    """Computes unigram counts and probabilities."""
    unigram_counts = Counter(tokens)
    corpus_size = sum(unigram_counts.values())
    unigram_probs = {word: count / corpus_size for word, count in unigram_counts.items()}
    return unigram_counts, unigram_probs, corpus_size

def compute_bigrams(tokens, unigram_counts):
    """Computes the frequency and probability of each bigram without smoothing."""
    bigram_counts = Counter(zip(tokens, tokens[1:]))  # Bigram frekanslarını hesapla

    bigram_probs = {} 

    # Her bir bigram için olasılığı hesapla: P(w2 | w1) = count(w1, w2) / count(w1)
    for (w1, w2), bigram_count in bigram_counts.items():
        prob = bigram_count / unigram_counts[w1]  # Olasılığı hesapla
        bigram_probs[(w1, w2)] = prob

    return bigram_counts, bigram_probs  # Frekanslar ve olasılıklar

def replace_least_freq_with_unkown(tokens, unigram_counts):
    """Replaces the least frequent word with UNK."""
    least_frequent = min(unigram_counts, key=unigram_counts.get)
    return ['UNK' if token == least_frequent else token for token in tokens]

def add_k_smoothing(bigram_counts, unigram_counts, k=0.5):
    """Computes smoothed bigram probabilities with Add-k smoothing."""
    vocabulary_size = len(unigram_counts)
    smoothed_probs = {}

    for (w1, w2), bigram_count in bigram_counts.items():
        prob = (bigram_count + k) / (unigram_counts[w1] + (k * vocabulary_size))
        smoothed_probs[(w1, w2)] = prob

    return smoothed_probs

def compute_sentence_probability(sentence, unigram_counts, bigram_counts,k=0.5):
    """Computes the probability of a sentence using bigram probabilities."""
    tokens = tokenize_and_tag_sentences(sentence)
    tokens = replace_with_unk(tokens, unigram_counts)

    prob = 1.0

    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        bigram_prob = get_bigram_probability(w1, w2, bigram_counts, unigram_counts, k)
        prob *= bigram_prob  # Multiply probabilities

    return prob

    return prob
def replace_with_unk(tokens, unigram_counts):
    """Replaces unknown tokens with UNK."""
    return [token if token in unigram_counts else 'UNK' for token in tokens]

def get_bigram_probability(w1, w2, bigram_counts, unigram_counts, k=0.5):
    """Computes the Add-k smoothed probability of a bigram."""
    vocabulary_size = len(unigram_counts)

    # Get the counts for the bigram and the first word
    bigram_count = bigram_counts.get((w1, w2), 0)
    w1_count = unigram_counts.get(w1, 0)

    # Apply Add-k smoothing
    prob = (bigram_count + k) / (w1_count +( k * vocabulary_size))
    return prob


def write_results_to_file(file_path, num_sentences, corpus_size, vocab_size, 
                          unigram_counts, unigram_probs, bigram_counts, 
                          bigram_probs, unigram_probs_with_unk, bigram_probs_with_unk, top_100_smoothed_bigrams, sentences_probs):
    """Writes the results to result.txt."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"Number of Sentences in File: {num_sentences}\n")
        file.write(f"Number of Total Tokens (Corpus Size): {corpus_size}\n")
        file.write(f"Number of Unique Words (Vocabulary Size): {vocab_size}\n\n")

        file.write("\nUnigrams Sorted wrt Frequencies (from Higher to Lower Frequencies):\n")
        for word, count in unigram_counts.most_common():
            file.write(f"{word}    {count}  {unigram_probs[word]}\n")

        file.write("\nBigrams Sorted wrt Frequencies (from Higher to Lower Frequencies):\n")
        for word, count in bigram_counts.most_common():
            file.write(f"{word}    {count}  {bigram_probs[word]}\n")


        file.write("\nAfter UNK addition and Smoothing Operations,\nTop 100 Bigrams wrt SmoothedProbability (from Higher to Lower):\n")
        for (w1, w2), smoothed_prob in top_100_smoothed_bigrams:
            file.write(f"({w1}, {w2})    {smoothed_prob}  {bigram_probs_with_unk[(w1, w2)]}\n")

        file.write("\nSentence Probabilities:\n")
        for sentence, prob in sentences_probs.items():
            file.write(f"{sentence}  {prob}\n")

def main():
    text = read_file(input("Enter the file path: "))

    num_sentences = count_sentences(text)

    tokens = tokenize_and_tag_sentences(text)

    unigram_counts, unigram_probs, corpus_size = compute_unigrams(tokens)
    bigram_counts, bigram_probs = compute_bigrams(tokens, unigram_counts)
    vocab_size = len(unigram_counts)

    # Replace the least frequent word with UNK
    tokens_with_unk = replace_least_freq_with_unkown(tokens, unigram_counts)
    unigram_counts_with_unk, unigram_probs_with_unk, _ = compute_unigrams(tokens_with_unk)
    bigram_counts_with_unk, bigram_probs_with_unk = compute_bigrams(tokens_with_unk, unigram_counts_with_unk)

    smoothed_bigram_probs = add_k_smoothing(bigram_counts_with_unk, unigram_counts_with_unk, k=0.5)

    top_100_smoothed_bigrams = sorted(smoothed_bigram_probs.items(), key=lambda x: x[1], reverse=True)[:100]

    #  Compute sentence probabilities
    sentences = [
        "Batuhan okula gitti.",
        "Batuhan eve geldi.",
    ]
    sentences_probs = {sentence: compute_sentence_probability(sentence, unigram_counts_with_unk, bigram_counts_with_unk, 0.5) for sentence in sentences}
    
    write_results_to_file(
        'result.txt', num_sentences, corpus_size, vocab_size,
        unigram_counts, unigram_probs, bigram_counts, bigram_probs ,unigram_probs_with_unk, bigram_probs_with_unk,
        top_100_smoothed_bigrams, sentences_probs
    )

if __name__ == "__main__":
    main()

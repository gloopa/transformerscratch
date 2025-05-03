import os
from tokenizers import Tokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


class TransformerTokenizer:
    """Wrapper for the tokenizer."""
    
    def __init__(self, tokenizer_path=None):
        """Initialize the tokenizer.
        
        Args:
            tokenizer_path: Path to the tokenizer directory
        """
        if tokenizer_path is not None:
            self.tokenizer = self._load_tokenizer(tokenizer_path)
        else:
            self.tokenizer = None
            
        # Special token IDs
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
    def _load_tokenizer(self, tokenizer_path):
        """Load the tokenizer.
        
        Args:
            tokenizer_path: Path to the tokenizer directory
            
        Returns:
            Tokenizer
        """
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        # Try to load from tokenizer.json first
        try:
            tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))
        except:
            # Fall back to loading from vocab and merges
            vocab_file = os.path.join(tokenizer_path, "vocab.json")
            merges_file = os.path.join(tokenizer_path, "merges.txt")
            
            if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
                raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_path}")
            
            tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        
        # Add post-processing for BOS/EOS tokens if needed
        tokenizer.post_processor = BertProcessing(
            ("<|eos|>", self.eos_token_id),
            ("<|bos|>", self.bos_token_id),
        )
        
        return tokenizer
    
    def train(self, files, vocab_size=50257, min_frequency=2, special_tokens=None):
        """Train the tokenizer.
        
        Args:
            files: List of files to train on
            vocab_size: Vocabulary size
            min_frequency: Minimum frequency for a token to be included
            special_tokens: List of special tokens
        """
        if special_tokens is None:
            special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
            
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(
            files=files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens
        )
        
        # Add post-processing for BOS/EOS tokens
        self.tokenizer.post_processor = BertProcessing(
            ("<|eos|>", self.eos_token_id),
            ("<|bos|>", self.bos_token_id),
        )
    
    def save(self, path):
        """Save the tokenizer.
        
        Args:
            path: Path to save the tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
            
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save(os.path.join(path, "tokenizer.json"))
        self.tokenizer.save_model(path)
    
    def encode(self, text, add_special_tokens=True, **kwargs):
        """Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            
        Returns:
            Encoded token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
            
        encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return encoded.ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
            
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self):
        """Get the vocabulary size.
        
        Returns:
            Vocabulary size
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
            
        return self.tokenizer.get_vocab_size()


def load_tokenizer(tokenizer_path):
    """Helper function to load a tokenizer.
    
    Args:
        tokenizer_path: Path to the tokenizer directory
        
    Returns:
        TransformerTokenizer
    """
    return TransformerTokenizer(tokenizer_path) 
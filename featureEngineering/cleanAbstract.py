import re
import unicodedata
from typing import Optional

def clean(
        text, 
        preserve_newlines = False,
        max_length = None,
        replacement_char= ' '):
    """
    Comprehensive text cleaning function for CSV compatibility.
    
    Args:
        text: Input text to clean
        preserve_newlines: If True, keeps newlines; if False, replaces with spaces
        max_length: Maximum length to truncate text (None for no limit)
        replacement_char: Character to use for replacements (default: space)
    
    Returns:
        Cleaned text safe for CSV usage
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Step 1: Fix encoding issues and smart quotes
    encoding_fixes = {
        # Smart quotes and their common encoding artifacts
        '‚Äú': '"',  # Opening smart quote artifact
        '‚Äù': '"',  # Closing smart quote artifact
        '‚Äô': "'",  # Apostrophe artifact
        '‚Äò': "'",  # Single quote artifact
        '‚Äî': '-',  # Em dash artifact
        '‚Äì': '-',  # En dash artifact
        '‚Ä¢': '•',  # Bullet artifact
        '√¢‚Ç¨': '"',  # Another quote artifact
        '√¢‚Ç¨': '"',  # Another quote artifact
        '√¢‚Ç¨': "'",  # Another apostrophe artifact
        
        # Direct smart quote replacements
        '"': '"',    # Left double quote
        '"': '"',    # Right double quote
        ''': "'",    # Left single quote
        ''': "'",    # Right single quote
        '—': '-',    # Em dash
        '–': '-',    # En dash
        '…': '...',  # Ellipsis
        '•': '*',    # Bullet point
        '‚': ',',    # Single low quote
        '„': '"',    # Double low quote
        '‹': '<',    # Single left angle quote
        '›': '>',    # Single right angle quote
        '«': '"',    # Left double angle quote
        '»': '"',    # Right double angle quote
    }
    
    for old, new in encoding_fixes.items():
        text = text.replace(old, new)
    
    # Step 2: Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Step 3: Handle problematic CSV characters
    if not preserve_newlines:
        # Replace all types of line breaks with spaces
        text = re.sub(r'\r\n|\r|\n', replacement_char, text)
    
    # Step 4: Fix quote issues
    # Remove unpaired quotes that could break CSV parsing
    # Count quotes and ensure they're balanced
    quote_count = text.count('"')
    if quote_count % 2 != 0:
        # If odd number of quotes, escape all internal quotes
        text = text.replace('"', '""')
    
    # Step 5: Handle other problematic characters
    problematic_chars = {
        '\x00': '',        # Null character
        '\x01': '',        # Start of heading
        '\x02': '',        # Start of text
        '\x03': '',        # End of text
        '\x04': '',        # End of transmission
        '\x05': '',        # Enquiry
        '\x06': '',        # Acknowledge
        '\x07': '',        # Bell
        '\x08': '',        # Backspace
        '\x0b': ' ',       # Vertical tab
        '\x0c': ' ',       # Form feed
        '\x0e': '',        # Shift out
        '\x0f': '',        # Shift in
        '\x10': '',        # Data link escape
        '\x11': '',        # Device control 1
        '\x12': '',        # Device control 2
        '\x13': '',        # Device control 3
        '\x14': '',        # Device control 4
        '\x15': '',        # Negative acknowledge
        '\x16': '',        # Synchronous idle
        '\x17': '',        # End of transmission block
        '\x18': '',        # Cancel
        '\x19': '',        # End of medium
        '\x1a': '',        # Substitute
        '\x1b': '',        # Escape
        '\x1c': '',        # File separator
        '\x1d': '',        # Group separator
        '\x1e': '',        # Record separator
        '\x1f': '',        # Unit separator
        '\x7f': '',        # Delete
    }
    
    for old, new in problematic_chars.items():
        text = text.replace(old, new)
    
    # Step 6: Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()  # Remove leading/trailing whitespace
    
    # Step 7: Remove or replace non-printable characters
    # Keep only printable ASCII + common extended characters
    text = ''.join(char if ord(char) >= 32 and ord(char) <= 126 or ord(char) in [9, 10, 13] or ord(char) >= 160 
                   else replacement_char for char in text)
    
    # Step 8: Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()  # Final whitespace cleanup
    
    # Step 9: Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0]  # Truncate at word boundary
        if len(text) < max_length - 3:
            text += '...'
    
    return text

#!/usr/bin/env python3
"""
Script para remover emojis problemáticos dos arquivos Python
"""

import re
import os

def clean_emojis_from_file(file_path):
    """Remove emojis and problematic unicode characters from a Python file"""
    
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Define emoji mappings for meaningful replacements
    emoji_replacements = {
        '🎯': '[TARGET]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '🔧': '[CONFIG]',
        '🏆': '[RESULT]',
        '📊': '[STATS]',
        '🦾': '[BIOMECH]',
        '🔬': '[ANALYSIS]',
        '⚠️': '[WARNING]',
        '🎾': '[TENNIS]',
        '🔹': ' -',
        '⚡': '[SPEED]',
        '⏱️': '[TIME]',
        '🚀': '[LAUNCH]',
        '💡': '[INFO]',
        '👍': '[GOOD]',
        '🔄': '[PROCESS]',
        '📁': '[FILE]',
        '🆕': '[NEW]',
        '🏗️': '[BUILD]',
        '💭': '[LOGIC]',
        '🤲': '[HAND]',
        '📈': '[METRICS]',
        '🥈': '[LEVEL2]',
        '🥉': '[LEVEL3]',
        '🧠': '[BRAIN]',
        '🎬': '[ACTION]'
    }
    
    # Replace specific emojis with meaningful text
    for emoji, replacement in emoji_replacements.items():
        content = content.replace(emoji, replacement)
    
    # Remove any remaining non-ASCII characters that might cause issues
    # Keep common accented characters but remove emojis and special symbols
    content = re.sub(r'[^\x00-\x7F\u00C0-\u017F]', '', content)
    
    # Write back the cleaned content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Cleaned emojis from: {file_path}")

def main():
    files_to_clean = [
        'improved_biomech_classifier_2d.py',
        'enhanced_racket_tracker_2d.py'
    ]
    
    for file_name in files_to_clean:
        if os.path.exists(file_name):
            print(f"Cleaning {file_name}...")
            clean_emojis_from_file(file_name)
        else:
            print(f"File not found: {file_name}")
    
    print("Emoji cleaning completed!")

if __name__ == "__main__":
    main()
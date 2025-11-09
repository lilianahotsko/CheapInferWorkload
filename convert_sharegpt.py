#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def convert_sharegpt(input_file: str, output_file: str = "datasets/sharegpt_full.jsonl", max_items: int = None):
    """Convert ShareGPT JSON to JSONL"""
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} conversations")
    
    if max_items:
        data = data[:max_items]
        print(f"Limiting to {max_items} items")
    
    converted = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            try:
                conversations = item.get('conversations', [])
                if len(conversations) >= 2:
                    # Get first user message and first assistant response
                    prompt = None
                    completion = None
                    
                    for conv in conversations:
                        if conv.get('from') == 'human' and prompt is None:
                            prompt = conv.get('value', '')
                        elif conv.get('from') == 'gpt' and completion is None:
                            completion = conv.get('value', '')
                        
                        if prompt and completion:
                            break
                    
                    if prompt and completion:
                        json.dump({'prompt': prompt, 'completion': completion}, f, ensure_ascii=False)
                        f.write('\n')
                        converted += 1
                        
                        if converted % 1000 == 0:
                            print(f"  Converted {converted} items...")
            
            except Exception as e:
                print(f"  Warning: Skipped item due to error: {e}")
                continue
    
    print(f"\n✓ Successfully converted {converted} items to {output_file}")
    
    # Show sample
    print("\nSample entry:")
    with open(output_file, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Completion: {sample['completion'][:100]}...")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_sharegpt.py <input_json_file> [max_items]")
        print("\nExample:")
        print("  python convert_sharegpt.py ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json")
        print("  python convert_sharegpt.py ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json 5000")
        sys.exit(1)
    
    input_file = sys.argv[1]
    max_items = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    convert_sharegpt(input_file, max_items=max_items)


#!/usr/bin/env python3
"""
Script to extract summaries from profile JSON files and save them with new numbering.
"""

import json
import os
from pathlib import Path

def extract_summaries(file_path):
    """Extract summaries from a profile JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summaries = {}
    # Skip metadata and process each profile
    for key, value in data.items():
        if key == "metadata":
            continue
        
        # Extract the summary if it exists
        if "Summary" in value:
            profile_index = value.get("Profile Index", 0)
            summaries[profile_index] = value["Summary"]
    
    return summaries

def main():
    # Define file paths
    input_file = Path("/home/zhou/persona/world_value_survey/gpt-4.1-mini_ca/data/all_profile_40.json")
    output_file = Path("/home/zhou/persona/world_value_survey/gpt-4.1-mini_ca/data/extracted_summaries_CAN.json")
    
    # Extract summaries from the file
    summaries_india = extract_summaries(input_file)
    
    # Renumber summaries
    combined_summaries = {}
    new_index = 1
    
    # Process India summaries
    for _, summary in sorted(summaries_india.items()):
        combined_summaries[new_index] = {
            "profile_id": new_index,
            "summary": summary
        }
        new_index += 1
    
    # Save the combined summaries to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_summaries, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted {len(combined_summaries)} summaries and saved to {output_file}")

if __name__ == "__main__":
    main()

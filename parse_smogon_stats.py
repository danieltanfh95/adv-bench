#!/usr/bin/env python3

import requests
import json
import re
import sys
from urllib.parse import urlparse
from pathlib import Path

def download_file(url):
    """Download the Smogon stats file from the given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)

def parse_filename_from_url(url):
    """Extract date and format info from URL to generate output filename."""
    # Example: https://www.smogon.com/stats/2025-07/moveset/gen3ou-1500.txt
    parsed = urlparse(url)
    path_parts = parsed.path.split('/')
    
    # Extract date (e.g., "2025-07")
    date_part = None
    format_part = None
    
    for i, part in enumerate(path_parts):
        if re.match(r'\d{4}-\d{2}', part):
            date_part = part
            if i + 2 < len(path_parts):
                format_part = Path(path_parts[i + 2]).stem
            break
    
    if not date_part or not format_part:
        print("Could not parse date and format from URL")
        sys.exit(1)
    
    return f"{date_part}-moveset-{format_part}.json"

def parse_smogon_stats(content):
    """Parse the Smogon stats text content and extract Pokemon data."""
    pokemon_data = {}
    lines = content.strip().split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for Pokemon name header - two consecutive separator lines followed by Pokemon name
        if (line.startswith('+') and i + 1 < len(lines) and 
            lines[i + 1].strip().startswith('+') and i + 2 < len(lines)):
            
            pokemon_line = lines[i + 2].strip()
            if pokemon_line.startswith('|') and pokemon_line.endswith('|'):
                # Extract Pokemon name
                pokemon_name = pokemon_line.strip('| ').strip()
                # Skip if this looks like a section header
                if (pokemon_name and not any(keyword in pokemon_name.lower() for 
                    keyword in ['raw count', 'avg. weight', 'abilities', 'items', 'spreads', 'moves', 'teammates', 'checks'])):
                    pokemon_data[pokemon_name] = {
                        'counters': [],
                        'partners': [],
                        'moves': []
                    }
                    i = parse_pokemon_section(lines, i + 3, pokemon_data[pokemon_name])
                    continue
        
        i += 1
    
    return pokemon_data

def parse_pokemon_section(lines, start_idx, pokemon_dict):
    """Parse a single Pokemon section and extract moves, teammates, and counters."""
    i = start_idx
    current_section = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for section end (next Pokemon - double separator lines)
        if (line.startswith('+') and i + 1 < len(lines) and 
            lines[i + 1].strip().startswith('+')):
            break
        
        # Identify sections
        if line == '| Moves' or 'Moves' in line and line.startswith('|') and line.endswith('|'):
            current_section = 'moves'
        elif line == '| Teammates' or 'Teammates' in line and line.startswith('|') and line.endswith('|'):
            current_section = 'teammates' 
        elif 'Checks and Counters' in line and line.startswith('|') and line.endswith('|'):
            current_section = 'counters'
        elif line.startswith('|') and current_section and not line.startswith('+'):
            # Parse data lines - exclude separator lines and section headers
            if current_section == 'moves':
                move_match = re.match(r'\|\s*([^|]+?)\s+[\d.]+%', line)
                if move_match:
                    move_name = move_match.group(1).strip()
                    if move_name and move_name != 'Other':
                        pokemon_dict['moves'].append(move_name)
            
            elif current_section == 'teammates':
                teammate_match = re.match(r'\|\s*([^|]+?)\s+[\d.]+%', line)
                if teammate_match:
                    teammate_name = teammate_match.group(1).strip()
                    if teammate_name:
                        pokemon_dict['partners'].append(teammate_name)
            
            elif current_section == 'counters':
                # Skip the percentage breakdown lines
                if '% KOed' in line or '% switched out' in line:
                    pass
                else:
                    counter_match = re.match(r'\|\s*([^|]+?)\s+[\d.]+', line)
                    if counter_match:
                        counter_name = counter_match.group(1).strip()
                        if counter_name:
                            pokemon_dict['counters'].append(counter_name)
        
        i += 1
    
    # Limit to top 10 for each category
    pokemon_dict['moves'] = pokemon_dict['moves'][:10]
    pokemon_dict['partners'] = pokemon_dict['partners'][:10]
    pokemon_dict['counters'] = pokemon_dict['counters'][:10]
    
    return i

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_smogon_stats.py <smogon_stats_url>")
        print("Example: python parse_smogon_stats.py https://www.smogon.com/stats/2025-07/moveset/gen3ou-1500.txt")
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Download the file
    print(f"Downloading {url}...")
    content = download_file(url)
    
    # Parse the content
    print("Parsing Smogon stats...")
    pokemon_data = parse_smogon_stats(content)
    
    # Generate output filename
    output_filename = parse_filename_from_url(url)
    
    # Write to JSON file
    print(f"Writing results to {output_filename}...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(pokemon_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully parsed {len(pokemon_data)} Pokemon and saved to {output_filename}")

if __name__ == "__main__":
    main()
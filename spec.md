# ADV OU AI Prediction Benchmark Specification

## Overview
This benchmark evaluates AI systems' ability to predict competitive Pokémon ADV OU (Advance Generation OverUsed) tier strategies by predicting the top 10 counters, partners, and movesets for given Pokémon.

## Data Sources
- **Primary Statistics**: Smogon usage statistics from `https://www.smogon.com/stats/2025-07/moveset/gen3ou-0.txt`
- **Context Data**: Pokémon data from Smogon Dex ADV OU format page
- **Ground Truth**: Historical competitive data and expert analysis

## Task Categories

### 1. Counter Prediction
**Objective**: Predict the top 10 Pokémon that effectively counter a given target Pokémon.

**Input**: 
- Target Pokémon name
- Optional: Current moveset context
- Optional: Team composition context

**Output Format**: Ordered list of 10 Pokémon names
```json
{
  "counters": [
    "Medicham",
    "Marowak",
    "Heracross",
    "Machamp",
    "Dugtrio",
    "Swampert",
    "Metagross",
    "Aerodactyl",
    "Claydol",
    "Magneton"
  ]
}
```

**Evaluation Metrics**:
- Top-k accuracy (k=1,3,5,10)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

### 2. Partner Prediction
**Objective**: Predict the top 10 Pokémon that synergize well as teammates with a given Pokémon.

**Input**:
- Target Pokémon name
- Optional: Known team members
- Optional: Team archetype (offense, balance, stall)

**Output Format**: Ordered list of 10 Pokémon names
```json
{
  "partners": [
    "Swampert",
    "Skarmory",
    "Gengar",
    "Zapdos",
    "Celebi",
    "Magneton",
    "Dugtrio",
    "Forretress",
    "Aerodactyl",
    "Claydol"
  ]
}
```

### 3. Move Prediction
**Objective**: Predict the top 10 most likely moves for a given Pokémon in ADV OU.

**Input**:
- Target Pokémon name
- Optional: Role specification (sweeper, wall, support)
- Optional: Team context

**Output Format**: Ordered list of moves
```json
{
  "moves": [
    "Rock Slide",
    "Earthquake", 
    "Hidden Power Bug",
    "Dragon Dance",
    "Fire Blast",
    "Pursuit",
    "Crunch",
    "Substitute",
    "Rest",
    "Sleep Talk"
  ]
}
```

## Benchmark Dataset Structure

### Primary Data Source
The benchmark uses Smogon usage statistics files parsed into JSON format:
```
data/
├── 2025-07-moveset-gen3ou-1500.json    # Main dataset from Smogon stats
├── pokemon_stats.json                  # Base stats, types, abilities for all ADV OU Pokemon  
├── type_chart.json                     # Type effectiveness chart
├── moves.json                          # Move data (power, accuracy, type, effect)
└── items.json                          # Item effects and usage
```

### Main Dataset Format
The primary dataset file (e.g., `2025-07-moveset-gen3ou-1500.json`) contains all Pokemon data:
```json
{
  "Tyranitar": {
    "counters": ["Medicham", "Marowak", "Heracross", "Machamp", "Dugtrio", "Swampert", "Metagross", "Aerodactyl", "Claydol", "Magneton"],
    "partners": ["Swampert", "Skarmory", "Gengar", "Zapdos", "Celebi", "Magneton", "Dugtrio", "Forretress", "Aerodactyl", "Claydol"],
    "moves": ["Rock Slide", "Earthquake", "Hidden Power Bug", "Dragon Dance", "Fire Blast", "Pursuit", "Crunch", "Substitute", "Rest", "Sleep Talk"]
  },
  "Metagross": {
    "counters": ["Moltres", "Charizard", "Blaziken", "Marowak", "Houndoom", "Flygon", "Typhlosion", "Dugtrio", "Arcanine", "Swampert"],
    "partners": ["Tyranitar", "Salamence", "Snorlax", "Suicune", "Zapdos", "Magneton", "Celebi", "Swampert", "Starmie", "Gengar"],
    "moves": ["Meteor Mash", "Explosion", "Earthquake", "Rock Slide", "Agility", "Protect", "Psychic", "Hidden Power Fire"]
  }
}
```

## Evaluation Framework

### LLM Evaluation Approach
This benchmark evaluates Large Language Models (LLMs) using the complete dataset as ground truth. No training/validation splits are needed since LLMs are evaluated in a zero-shot or few-shot manner using their existing knowledge.

### Test Set
- **Full Dataset**: All Pokémon in the parsed JSON file serve as test cases
- **Evaluation Method**: Each Pokémon is presented to the LLM as a prediction task
- **Context Variations**: Test with different context levels (no context, partial team, specific roles)

### Baseline Comparisons
1. **Random Baseline**: Random selection from all viable ADV OU Pokémon/moves
2. **Usage-Based Baseline**: Predictions based purely on overall usage statistics
3. **Type-Effectiveness Baseline**: Predictions based on type matchups alone
4. **Human Expert Baseline**: Predictions from experienced competitive players

### Success Criteria
- **Counter Prediction**: Top-5 accuracy > 60%
- **Partner Prediction**: Top-5 accuracy > 50%
- **Move Prediction**: Top-3 accuracy > 40%

## Implementation Requirements

### Input Processing
- Parse Pokémon names (handle variations, abbreviations)
- Extract relevant context from team compositions
- Normalize stat and usage data

### Context Integration
- Incorporate meta-game knowledge
- Consider team archetypes and roles
- Account for item and ability interactions

### Output Generation
- Ensure valid Pokémon/move combinations
- Return predictions in ranked order (most likely first)

## Scoring System

### Primary Score (Weighted Average)
- Counter Prediction: 40% weight
- Partner Prediction: 35% weight  
- Move Prediction: 25% weight

### Bonus Points
- Correct identification of niche counters (+5%)
- Accurate prediction of uncommon but viable sets (+3%)
- Consistent performance across different Pokémon archetypes (+2%)

## Data Format Examples

### Sample Input
```json
{
  "target_pokemon": "Tyranitar",
  "context": {
    "known_moves": ["Rock Slide", "Earthquake"],
    "team_members": ["Skarmory", "Gengar"],
    "format": "ADV OU"
  }
}
```

### Sample Expected Output
```json
{
  "counters": ["Medicham", "Marowak", "Heracross", "Machamp", "Dugtrio", "Swampert", "Metagross", "Aerodactyl", "Claydol", "Magneton"],
  "partners": ["Swampert", "Skarmory", "Gengar", "Zapdos", "Celebi", "Magneton", "Dugtrio", "Forretress", "Aerodactyl", "Claydol"],
  "moves": ["Rock Slide", "Earthquake", "Hidden Power Bug", "Dragon Dance", "Fire Blast", "Pursuit", "Crunch", "Substitute", "Rest", "Sleep Talk"]
}
```

## Additional Considerations

### Meta-Game Evolution
- Account for shifts in the competitive meta
- Handle emergence of new strategies
- Adapt to usage pattern changes

### Edge Cases
- Handle Pokémon with multiple viable roles
- Consider unconventional strategies
- Account for team-specific adaptations

### Validation Methods
- Cross-validation with expert player rankings
- Comparison against tournament results
- A/B testing with competitive community

## Success Metrics Summary
- **Accuracy**: Percentage of correct top-k predictions
- **Coverage**: Ability to make reasonable predictions for all viable Pokémon
- **Novelty**: Identification of lesser-known but effective strategies
- **Consistency**: Stable performance across different meta conditions
- **Explainability**: Quality of reasoning provided for predictions
import json

import pandas as pd
import tensorflow as tf
from model import Word2Vec
from tabulate import tabulate

with open("../data/eval_data_poke_and_types.json", "r") as f:
    POKEMONS_BY_TYPE = json.load(f)

def test_type_similarity(model_path="./saved_model"):
    """Test if pokemons are closer to their type than to any other type."""
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, custom_objects={"Word2Vec": Word2Vec})
    
    print("Starting tests...")
    results = []
    for p_type, pokemons in POKEMONS_BY_TYPE.items():
        total_comparisons = 0
        correct_comparisons = 0
        pokemon_count = 0
        
        for pokemon in pokemons:
            if pokemon not in model.vocabulary:
                continue
                
            pokemon_count += 1
            close_similarity = model.compute_similarity(pokemon, p_type)
            other_types = [t for t in POKEMONS_BY_TYPE if t != p_type]
            
            for other_type in other_types:
                far_similarity = model.compute_similarity(pokemon, other_type)
                if close_similarity > far_similarity:
                    correct_comparisons += 1
                total_comparisons += 1
            
        if total_comparisons > 0:
            percentage = 100.0 * correct_comparisons / total_comparisons
            results.append({
                "Poke type": p_type,
                "Poke count": pokemon_count,
                "Tests": total_comparisons,
                "Correct": correct_comparisons,
                "Percentage": round(percentage, 1),
            })
    
    print("\nFinal summary...")
    df = pd.DataFrame(results)
    df = df.sort_values("Percentage", ascending=False)
    print(tabulate(df, headers="keys", tablefmt="pretty", floatfmt=".2f"))
    avg_row = {
        "Type": "AVERAGE",
        "Pokemon count": df["Poke count"].sum(),
        "Tests": df["Tests"].sum(),
        "Correct": df["Correct"].sum(),
        "Percentage": round(
            100.0 * df["Correct"].sum() / df["Tests"].sum() if df["Tests"].sum() > 0 else 0, 1
        ),
    }   
    print("\nOverall Statistics:")
    print(f"Pokemons tested: {avg_row['Pokemon count']}")
    print(f"Total comparisons: {avg_row['Tests']}")
    print(f"Correct comparisons: {avg_row['Correct']}")
    print(f"Overall accuracy: {avg_row['Percentage']:.2f}%")

if __name__ == "__main__":
    test_type_similarity()

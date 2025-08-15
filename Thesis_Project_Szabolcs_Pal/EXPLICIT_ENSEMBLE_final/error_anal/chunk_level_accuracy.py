from pathlib import Path
import matplotlib.pyplot as plt
def analyze_chunkwise_correct_predictions(df, output_path):
    """
    Creates a plot showing total correct predictions per chunk index across all qid_bases.
    
    Parameters:
    - df (pd.DataFrame): Must contain ['qid', 'label', 'pred']
    - output_path (str or Path): Path to save the plot
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import re

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract chunk_id from qid
    df["chunk_id"] = df["qid"].apply(lambda x: re.findall(r'chunk\d+', x)[0] if "chunk" in x else "chunk0")

    # Mark correct predictions
    df["correct"] = df["label"] == df["pred"]

    # Count correct predictions per chunk_id
    chunk_correct_counts = df[df["correct"]].groupby("chunk_id").size().sort_index()

    # Plot
    plt.figure(figsize=(10, 5))
    chunk_correct_counts.plot(kind="bar")
    plt.title("Total Correct Predictions per Chunk Index")
    plt.xlabel("Chunk ID")
    plt.ylabel("Number of Correct Predictions")
    plt.tight_layout()
    plt.savefig(output_path / "correct_predictions_per_chunk.png")
    plt.close()


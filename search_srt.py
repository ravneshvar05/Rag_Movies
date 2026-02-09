
import re

file_path = r"d:\Intership Projects\Rag_Movies\movie_rag\data\raw_srt\K.G.F Chapter 1 (2018) (NetNaija.com)-en.srt"
output_path = r"d:\Intership Projects\Rag_Movies\movie_rag\srt_search_results.txt"

keywords = ["Andrews", "Bangalore", "Bengaluru", "Garuda", "kill", "offer", "deal", "come"]

print(f"Searching in {file_path}...")

try:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
except UnicodeDecodeError:
    print("UTF-8 failed, trying latin-1")
    with open(file_path, "r", encoding="latin-1") as f:
        lines = f.readlines()

with open(output_path, "w", encoding="utf-8") as out:
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                out.write(f"Line {i+1}: {line.strip()}\n")
                # Print context
                start = max(0, i-5)
                end = min(len(lines), i+6)
                for j in range(start, end):
                    out.write(f"  {lines[j].strip()}\n")
                out.write("-" * 20 + "\n")
                break # Avoid duplicate printing for same line

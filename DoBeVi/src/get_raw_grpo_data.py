import os
import json
import argparse
from search.search_tree import get_data_for_grpo

def main():
    parser = argparse.ArgumentParser(description="Process tree JSON files and generate raw GRPO data.")
    parser.add_argument("--tree_dir", type=str, required=True, help="Path to the directory containing tree JSON files.")
    args = parser.parse_args()

    tree_dir = os.path.abspath(args.tree_dir)
    parent_dir = os.path.dirname(tree_dir)
    raw_data_dir = os.path.join(parent_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    json_files = [os.path.join(tree_dir, f) for f in os.listdir(tree_dir) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files in {tree_dir}")

    for json_file in json_files:
        file_name = os.path.basename(json_file)
        theorem_name = file_name.split("_tree")[0].split("_")[-1]

        with open(json_file, "r", encoding="utf-8") as f:
            tree_data = json.load(f)

        # Fill in missing fields if not present
        if "repo_path" not in tree_data:
            tree_data["repo_path"] = "/data0/xs/LLM-ATP/dataset/FineLeanCorpus-lean"
        if "file_path" not in tree_data:
            tree_data["file_path"] = f"FineLeanCorpusLean/{theorem_name}.lean"
        if "theorem_name" not in tree_data:
            tree_data["theorem_name"] = theorem_name

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=4)

        # Generate GRPO data
        try:
            raw_data = get_data_for_grpo(json_file)
        except Exception as e:
            print(f"❌ Failed to process {json_file}: {e}")
            continue

        output_file = os.path.join(raw_data_dir, file_name.replace(".json", "_raw.json"))
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Raw data saved to {output_file}")

if __name__ == "__main__":
    main()

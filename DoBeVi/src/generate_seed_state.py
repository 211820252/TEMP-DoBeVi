import os
import json
from search.search_tree import load_tree, choose_good_nodes

# tree json 所在的文件夹目录
tree_dir = "/data1/xs/ATP/results/FineLeanCorpus/results_20251020_113938/tree"
parent_dir = os.path.dirname(tree_dir)
seed_state_dir = os.path.join(parent_dir, "seed_state")
os.makedirs(seed_state_dir, exist_ok=True)

json_files = [os.path.join(tree_dir, f) for f in os.listdir(tree_dir) if f.endswith('.json')]

for json_file in json_files:
    file_name = os.path.basename(json_file)
    theorem_id = file_name.split('_tree')[0].split('_')[-1]
    try:
        root_node = load_tree(json_file)
    except Exception as e:
        print(f"❗ Failed to load tree: {json_file}")
        print(f"❗ Error: {e}")
        
    good_nodes = choose_good_nodes(root = root_node, theorem_id=theorem_id)
    
    output_file = os.path.join(seed_state_dir, file_name.replace('.json', '_seed_state.json'))
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(good_nodes, f, ensure_ascii=False, indent=4)
    
    print(f"Good nodes saved to {output_file}")
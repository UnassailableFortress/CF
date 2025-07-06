import json
import os

def split_json_lines_file(input_path, output_prefix, chunk_size=500000):
    chunk = []
    part = 0
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    with open(input_path, 'r') as infile:
        for i, line in enumerate(infile):
            if line.strip():
                chunk.append(line)
            if len(chunk) == chunk_size:
                out_path = f'{output_prefix}_part{part}.json'
                with open(out_path, 'w') as out:
                    out.writelines(chunk)
                print(f"Saved: {out_path} ({len(chunk)} lines)")
                part += 1
                chunk = []
        # Save remaining lines
        if chunk:
            out_path = f'{output_prefix}_part{part}.json'
            with open(out_path, 'w') as out:
                out.writelines(chunk)
            print(f"Saved final: {out_path} ({len(chunk)} lines)")

# Customize this
if __name__ == "__main__":
    input_file = "/home/zalert_rig305/Desktop/EE/Programs/Clothing_Shoes_and_Jewelry.json"
    output_prefix = "/home/zalert_rig305/Desktop/EE/Programs/Clothing_Split/part"
    split_json_lines_file(input_file, output_prefix, chunk_size=500000)

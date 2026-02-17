import hashlib
import json

def dedup_data(data_list):
    """
    Hashes data samples and removes duplicates.
    """
    print(f"🧹 Deduplicating {len(data_list)} items...")
    seen_hashes = set()
    unique_data = []
    
    for item in data_list:
        # Create a hash of the content
        content_str = json.dumps(item, sort_keys=True)
        content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_data.append(item)
    
    removed = len(data_list) - len(unique_data)
    print(f"✅ Removed {removed} duplicates. Retained {len(unique_data)} unique items.")
    return unique_data

# DEMO
if __name__ == "__main__":
    # Simulated Dataset with duplicates
    raw_data = [
        {"id": 1, "text": "Hello World"},
        {"id": 2, "text": "Machine Learning is cool"},
        {"id": 3, "text": "Hello World"}, # Duplicate
        {"id": 4, "text": "Green AI"},
        {"id": 5, "text": "Machine Learning is cool"}, # Duplicate
    ]
    
    dedup_data(raw_data)

import os
import time

# CONFIG
ARTIFACT_DIR = "./checkpoints" # Ensure this exists
MAX_AGE_DAYS = 90
MAX_AGE_SECONDS = 5 # Set to 5 seconds for DEMO purposes

def cleanup_stale_files():
    print(f"--- 🗑️ Cleaning up files older than {MAX_AGE_SECONDS} seconds (Demo Mode) ---")
    
    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR)
        # Create a dummy old file
        with open(os.path.join(ARTIFACT_DIR, "old_model.pt"), "w") as f:
            f.write("dummy data")
        print("Created dummy file for testing.")
        time.sleep(6) # Wait for it to become 'stale'

    now = time.time()
    deleted_count = 0
    
    for filename in os.listdir(ARTIFACT_DIR):
        filepath = os.path.join(ARTIFACT_DIR, filename)
        if os.path.isfile(filepath):
            file_age = now - os.path.getmtime(filepath)
            if file_age > MAX_AGE_SECONDS:
                os.remove(filepath)
                print(f"Deleted Stale Artifact: {filename}")
                deleted_count += 1
                
    print(f"✅ Cleanup Complete. Removed {deleted_count} files.")

if __name__ == "__main__":
    cleanup_stale_files()

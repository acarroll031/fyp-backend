import requests
import os
import time

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"
MODULE_CODE = "CS161"  # Change this to match your target module
CSV_FOLDER = "Sample Assessment CSVs"  # Folder where your 12 files are
TOTAL_ASSESSMENTS = 12


def seed_assessments():
    print(f"🚀 Starting automated upload for module: {MODULE_CODE}")

    # Optional: Login if you add authentication to the upload endpoint later
    # token = login_and_get_token()
    # headers = {"Authorization": f"Bearer {token}"}

    for i in range(1, TOTAL_ASSESSMENTS + 1):
        file_name = f"assessment_{i}.csv"
        file_path = os.path.join(CSV_FOLDER, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}. Skipping.")
            continue

        # Calculate progress (e.g., 1/12 = 0.08, 12/12 = 1.0)
        progress = round(i / TOTAL_ASSESSMENTS, 2)

        print(f"📤 Uploading {file_name} (Progress: {progress})...")

        try:
            # Open the file in binary mode
            with open(file_path, "rb") as f:
                files = {"file": (file_name, f, "text/csv")}

                # Send the POST request
                # Note: progress_in_semester is a query parameter
                response = requests.post(
                    f"{API_URL}/students/{MODULE_CODE}/grades",
                    params={"progress_in_semester": progress},
                    files=files
                )

            if response.status_code == 200:
                print(f"✅ Success: {file_name} processed.")
            else:
                print(f"❌ Failed: {file_name} - {response.text}")

        except Exception as e:
            print(f"❌ Error uploading {file_name}: {e}")

        # Optional: Add a small delay to simulate real-time (helps see the history order clearly)
        time.sleep(0.5)

    print("\n✨ All uploads complete!")


if __name__ == "__main__":
    seed_assessments()
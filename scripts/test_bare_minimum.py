import os

def test_run():
    print("=== CONTAINER STARTED SUCCESSFULLY ===")
    try:
        files = os.listdir('/opt/ml/processing/input/')
        print(f"Files found in input directory: {files}")
    except Exception as e:
        print(f"Error reading directory: {e}")
    print("=== CONTAINER FINISHED SUCCESSFULLY ===")

if __name__ == '__main__':
    test_run()
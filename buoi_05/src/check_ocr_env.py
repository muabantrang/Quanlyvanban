import importlib
import sys

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        return "PASS"
    except ImportError:
        return "FAIL"

def main():
    print(f"{'Công cụ / Thư viện':<20} | {'Trạng thái':<10}")
    print("-" * 35)
    
    packages = {
        "Python": sys.version.split()[0],
        "PyMuPDF": "fitz",
        "Pillow": "PIL",
        "Llama_cloud": "llama_cloud",
        "Pydantic": "pydantic",
        "Streamlit": "streamlit",
        "python-dotenv": "dotenv"
    }

    print(f"{'Python':<20} | PASS ({sys.version.split()[0]})")
    for pkg, imp_name in list(packages.items())[1:]:
        status = check_package(pkg, imp_name)
        print(f"{pkg:<20} | {status:<10}")

if __name__ == "__main__":
    main()

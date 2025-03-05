import pickle
import os

def view_cookies():
    """Load and print the contents of the cookies, localStorage, and sessionStorage files."""
    cookies_file = 'cookies.pkl'
    local_storage_file = 'local_storage.pkl'
    session_storage_file = 'session_storage.pkl'

    # Load and print cookies if the file exists
    if os.path.exists(cookies_file):
        print(f"\nCOOKIES:\n")
        with open(cookies_file, 'rb') as file:
            cookies = pickle.load(file)
            if cookies:
                for cookie in cookies:
                    print(cookie)  # Print each cookie in the file
            else:
                print("No cookies found.")
    else:
        print(f"{cookies_file} does not exist.")

    # Load and print localStorage cookies if the file exists
    if os.path.exists(local_storage_file):
        print(f"\nLOCAL_STORAGE COOKIES:\n")
        with open(local_storage_file, 'rb') as file:
            local_storage_cookies = pickle.load(file)
            if local_storage_cookies:
                for cookie in local_storage_cookies:
                    print(cookie)  # Print each cookie in the file
            else:
                print("No localStorage cookies found.")
    else:
        print(f"{local_storage_file} does not exist.")

    # Load and print sessionStorage cookies if the file exists
    if os.path.exists(session_storage_file):
        print(f"\nSESSION_STORAGE COOKIES:\n")
        with open(session_storage_file, 'rb') as file:
            session_storage_cookies = pickle.load(file)
            if session_storage_cookies:
                for cookie in session_storage_cookies:
                    print(cookie)  # Print each cookie in the file
            else:
                print("No sessionStorage cookies found.")
    else:
        print(f"{session_storage_file} does not exist.")

# Call the function to view the cookies
view_cookies()

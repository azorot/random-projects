from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import pickle
import time

# Path to your Firefox profile
profile_path = '/home/azoroth/.mozilla/firefox/gt4dwpe1.default-release'

# Setup Firefox options and profile
options = Options()
options.add_argument(f'-profile {profile_path}')

# Initialize the WebDriver
driver = webdriver.Firefox(options=options)

def random_sleep(min_seconds=2, max_seconds=5):
    """Sleep for a random amount of time to simulate human behavior."""
    sleep_time = random.uniform(min_seconds, max_seconds)
    print(f"Sleeping for {sleep_time:.2f} seconds")
    time.sleep(sleep_time)

def check_logged_in():
    """Check if the user is logged in by inspecting the page for specific elements."""
    driver.get('https://phemex.com')
    random_sleep()

    # Check for user-specific element or login button
    if "Login" in driver.page_source:  # You can adjust this based on your observation of the Phemex page
        print("Not logged in.")
        return False
    else:
        print("Already logged in.")
        return True

# Main function to check login state and load cookies
def main():
    if not check_logged_in():
        print("Logging in...")
        # If not logged in, you can execute your login code here
        driver.get('https://phemex.com/login')
        random_sleep()

        # Enter your login details here (if needed)
        # email_input = driver.find_element_by_name('email')
        # password_input = driver.find_element_by_name('password')
        # email_input.send_keys('your_email')
        # password_input.send_keys('your_password')
        # password_input.submit()

        # After login, save cookies, etc.
        # save_cookies()  # Save cookies if you want to use them later

    else:
        print("User is already logged in.")

if __name__ == "__main__":
    main()

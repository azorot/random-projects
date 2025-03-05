import pickle
import time
from selenium import webdriver

# Set up the Firefox WebDriver
driver = webdriver.Firefox()

# Navigate to the website
driver.get('https://phemex.com/login')

# Perform login actions manually or automate them as needed
# For simplicity, you can manually log in, or use the login functionality from earlier in the script

time.sleep(100)  # Wait for the login to complete (adjust as needed)

# Save cookies after login
cookies = driver.get_cookies()

# Save cookies to a file (pickle format)
with open('cookies.pkl', 'wb') as file:
    pickle.dump(cookies, file)

# Save localStorage
local_storage = driver.execute_script("return window.localStorage;")

# Save localStorage to a file (pickle format)
with open('local_storage.pkl', 'wb') as file:
    pickle.dump(local_storage, file)

# Save sessionStorage
session_storage = driver.execute_script("return window.sessionStorage;")

# Save sessionStorage to a file (pickle format)
with open('session_storage.pkl', 'wb') as file:
    pickle.dump(session_storage, file)

print("Cookies, localStorage, and sessionStorage saved.")

# Close the browser
driver.quit()

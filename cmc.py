# https://coinmarketcap.com/currencies/terrausd/
# https://coinmarketcap.com/currencies/terra-luna/

import requests
from bs4 import BeautifulSoup

def get_crypto_price(url, selector):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        price_element = soup.select_one(selector)
        if price_element:
            price_text = price_element.text.strip()
            # Remove non-numeric characters like '$' and ','
            price_value = float(''.join(c for c in price_text if c.isdigit() or c == '.'))
            return price_value
        else:
            return None  # Price element not found
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except ValueError:
        print("Error parsing price value.")
        return None

# URLs for TerraUSD and Terra Luna Classic
terrausd_url = "https://coinmarketcap.com/currencies/terrausd/"
terraluna_url = "https://coinmarketcap.com/currencies/terra-luna/"

# CSS selector based on user's provided HTML snippet
price_selector = 'span[data-test="text-cdp-price-display"]'

# Get prices
terrausd_price = get_crypto_price(terrausd_url, price_selector)
terraluna_price = get_crypto_price(terraluna_url, price_selector)

# Store prices in variables
if terrausd_price is not None:
    usd_price = terrausd_price
    print(f"TerraUSD (USD) Price: ${usd_price}")
else:
    print("Could not retrieve TerraUSD price.")

if terraluna_price is not None:
    luna_price = terraluna_price
    print(f"Terra Luna Classic (LUNC) Price: ${luna_price:.8f}") # Assuming Terra Luna refers to Terra Luna Classic (LUNC)
else:
    print("Could not retrieve Terra Luna Classic price.")

# Example calculation (you can add your calculations here)
# if terrausd_price is not None and terraluna_price is not None:
#     total_price = terrausd_price + terraluna_price
#     print(f"Sum of prices: ${total_price}")
#ratio calculation
ustc = usd_price * 100
print(f'percentage of USTC: {ustc}%')

#calculate theoretical price of lunc if ustc reaches 1$
lunc= luna_price

cross_multi = (lunc * 100)/ustc
print(f'Theoretical price of LUNC if USTC reaches 1$: {cross_multi}')
mylunc = 3487471
profit = mylunc * cross_multi

print(f'profit: ${profit:,.2f}')
from camoufox.sync_api import Camoufox

inf = []
prices = []

with Camoufox() as browser:
    page = browser.new_page()
    page.goto('https://torob.com/')
    page.click('input#search-query-input')
    page.keyboard.type("گوشی")
    page.wait_for_timeout(5000)
    page.keyboard.press('Enter')
    page.wait_for_timeout(5000)
    for i in page.query_selector_all('h2.ProductCard_desktop_product-name__JwqeK'):
        inf.append(i.inner_text())
    for i in page.query_selector_all('div.ProductCard_desktop_product-price-text__y20OV'):
        prices.append(i.inner_text())
    
li = list(zip(inf, prices))
lines = [f"{inf} | {prices}" for inf, prices in li]
f =  "\n".join(lines)
print(f)
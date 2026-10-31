import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


PROJECT_DIR = Path(__file__).resolve().parent.parent
CATALOGUE_PATH = PROJECT_DIR / "products.json"

ALLOWED_HOSTS = {
    "koreanbeautypoint.com",
    "www.koreanbeautypoint.com",
}


def allowed_page(url):
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in ALLOWED_HOSTS
        and parsed.username is None
        and parsed.password is None
        and parsed.port in {None, 443}
    )


def fetch_html(session, url):
    current = url

    for _ in range(5):
        if not allowed_page(current):
            raise ValueError("Not an allowed Korean Beauty Point URL.")

        response = session.get(
            current,
            timeout=(10, 20),
            allow_redirects=False,
        )

        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")

            if not location:
                raise ValueError("Redirect has no destination.")

            current = urljoin(current, location)
            continue

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" not in content_type:
            raise ValueError("Page did not return HTML.")

        return response.text, current

    raise ValueError("Too many redirects.")


def find_image(html, page_url):
    soup = BeautifulSoup(html, "html.parser")

    selectors = [
        'meta[property="og:image:secure_url"]',
        'meta[property="og:image"]',
        'meta[name="twitter:image"]',
        'meta[itemprop="image"]',
        'img[itemprop="image"]',
        'img#product_detail_main_image',
        'img.product_detail_img',
    ]

    for selector in selectors:
        for element in soup.select(selector):
            candidate = (
                element.get("content")
                or element.get("src")
                or element.get("data-src")
            )

            if not candidate:
                continue

            image_url = urljoin(page_url, candidate)
            parsed = urlparse(image_url)

            if parsed.scheme == "https" and parsed.hostname:
                return image_url

    return ""


def main():
    original = CATALOGUE_PATH.read_text(encoding="utf-8")
    products = json.loads(original)

    if not isinstance(products, list):
        raise SystemExit("products.json must contain a JSON list.")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "SkinAlyzeCatalogueHelper/1.0",
        "Accept": "text/html",
    })

    changed = 0

    for product in products:
        if not isinstance(product, dict):
            continue

        name = product.get("name", "Unnamed product")
        url = str(product.get("url") or "").strip()

        if product.get("image"):
            print(f"Keeping existing image: {name}")
            continue

        if not allowed_page(url):
            print(f"Skipped invalid retailer URL: {name}")
            continue

        try:
            html, final_url = fetch_html(session, url)
            image_url = find_image(html, final_url)

            if image_url:
                product["image"] = image_url
                changed += 1
                print(f"\nImage candidate found: {name}")
                print(image_url)
            else:
                print(f"\nNo image metadata found: {name}")

        except (requests.RequestException, ValueError) as error:
            print(f"\nCould not extract image: {name}")
            print(error)

        time.sleep(1)

    if not changed:
        print("\nNo images added. products.json is unchanged.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = PROJECT_DIR / f"products_backup_{stamp}.json"
    backup_path.write_text(original, encoding="utf-8")

    temporary_path = PROJECT_DIR / "products.pending.json"
    temporary_path.write_text(
        json.dumps(products, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(CATALOGUE_PATH)

    print(f"\nAdded {changed} image candidates.")
    print(f"Backup saved: {backup_path.name}")
    print("Updated products.json.")
    print("Check that each image shows the correct product.")


if __name__ == "__main__":
    main()
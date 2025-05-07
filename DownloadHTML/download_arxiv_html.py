import os
import requests
from bs4 import BeautifulSoup

class HTML_Download:
    def __init__(self, url, output_dir=None):
        self.url = url
        self.output_dir = output_dir

        self._create_output_dir()


    def _create_output_dir(self):
        if self.output_dir is None:
            self.output_dir = self.url.split("/")[-1].replace(".", "_")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        

    # def download_html(self):
    #     try:
    #         print("[*] Downloading HTML")
    #         response = requests.get(self.url)
    #         html_file = os.path.join(self.output_dir, "paper.html")
    #         with open(html_file, "w", encoding='utf-8') as f:
    #             f.write(response.text)
    #         print("[+]")
    #     except Exception as e:
    #         print(f"[-] Error: {e}")

    def download_html(self):
        try:
            print("[*] Downloading HTML")
            response = requests.get(self.url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove <base> tag if present
            base_tag = soup.find('base')
            if base_tag:
                base_tag.decompose()

            # Rewrite image src attributes
            for img_tag in soup.find_all('img'):
                src = img_tag.get('src')
                if src:
                    img_filename = src.split('/')[-1]
                    img_tag['src'] = img_filename

            html_file = os.path.join(self.output_dir, "paper.html")
            with open(html_file, "w", encoding='utf-8') as f:
                f.write(str(soup))
            print("[+]")
        except Exception as e:
            print(f"[-] Error: {e}")


    def download_images(self):
        try:
            print("[*] Downloading Images")
            # Send a GET request to the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(self.url, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all <img> tags
            img_tags = soup.find_all('img')

            if not img_tags:
                print(f"No <img> tags found on {self.url}")
                return

            print(f"Found {len(img_tags)} image tags. Attempting to download...")

            for img_tag in img_tags:
                img_url = img_tag.get('src')
                if not img_url:
                    # Sometimes the actual URL is in 'data-src' or other attributes
                    img_url = img_tag.get('data-src')

                if img_url:
                    # Make sure the URL is absolute
                    img_url = self.url + '/' + img_url
                    try:
                        # Get the image content
                        img_response = requests.get(img_url, headers=headers, stream=True)
                        img_response.raise_for_status()

                        # Extract image name from URL
                        img_name = os.path.join(self.output_dir, img_url.split('/')[-1].split('?')[0]) # Basic name cleaning

                        # Ensure the image name has an extension, default to .png if not obvious
                        if not os.path.splitext(img_name)[1]:
                            content_type = img_response.headers.get('content-type')
                            if content_type and 'image' in content_type:
                                extension = content_type.split('/')[-1]
                                img_name += f".{extension}"
                            else:
                                img_name += ".png" # Default extension

                        # Save the image
                        with open(img_name, 'wb') as f:
                            for chunk in img_response.iter_content(1024):
                                f.write(chunk)
                        print(f"Successfully downloaded: {img_name}")

                    except requests.exceptions.RequestException as e:
                        print(f"Could not download {img_url}. Error: {e}")
                    except Exception as e:
                        print(f"An error occurred for {img_url}. Error: {e}")
                else:
                    print("Found an img tag without a 'src' or 'data-src' attribute.")
            print("[+]")
        except Exception as e:
            print(f"[-] Error: {e}")


if __name__ == "__main__":
    
    HTML_URL = "https://arxiv.org/html/2404.12253v2"
    OUTPUT_DIR = None

    html_download = HTML_Download(url=HTML_URL, output_dir=OUTPUT_DIR)
    html_download.download_html()
    html_download.download_images()
   
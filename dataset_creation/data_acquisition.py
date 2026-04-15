import os
import logging
import time
from icrawler.builtin import BingImageCrawler
from config_data import CLASS_MAPPINGS, DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_images():
    """
    Downloads images using BingImageCrawler based on defined keyword mappings.
    Incorporates basic rate limiting by sleeping between queries.
    """
    logging.info(f"Starting data acquisition. Raw data directory: {DATA_DIR}")
    for class_name, queries in CLASS_MAPPINGS.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        logging.info(f"Processing class: {class_name}")
        
        for q in queries:
            logging.info(f"Querying: {q}")
            
            # Using Bing
            crawler = BingImageCrawler(
                storage={'root_dir': class_dir},
                log_level=logging.WARNING
            )
            
            try:
                # Scrape images properly dividing 500 max by number of queries
                crawler.crawl(keyword=q, max_num=500 // len(queries))
                
                import uuid
                for filename in os.listdir(class_dir):
                    if filename.startswith("0000"):
                        old_path = os.path.join(class_dir, filename)
                        new_path = os.path.join(class_dir, f"{uuid.uuid4().hex}.jpg")
                        try:
                            os.rename(old_path, new_path)
                        except:
                            pass
            except Exception as e:
                logging.error(f"Error crawling for '{q}': {e}")
                
            logging.info(f"Finished query '{q}'. Sleeping for rate limiting...")
            time.sleep(3) # Small wait to avoid rate limiting
            
    logging.info("Data acquisition complete.")

if __name__ == "__main__":
    download_images()

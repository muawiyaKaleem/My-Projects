# step1_improved_scraper.py
import requests
from bs4 import BeautifulSoup
import json
import time
import csv
from datetime import datetime
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import random

class YouTubeScraper:
    def __init__(self):
        # List of different Chrome versions for rotation
        self.chrome_user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        ]
        self.current_agent_index = 0
        self.comments = []
        self.driver = None
        
    def setup_driver(self):
        """
        Setup Chrome WebDriver with stealth options
        # Configures Chrome browser with anti-detection settings
        # Uses headless mode for faster execution
        # Applies random user agent rotation for better success rate
        """
        chrome_options = Options()
        
        # Stealth options to avoid detection
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Random user agent
        user_agent = self.chrome_user_agents[self.current_agent_index]
        chrome_options.add_argument(f"--user-agent={user_agent}")
        self.current_agent_index = (self.current_agent_index + 1) % len(self.chrome_user_agents)
        
        # Additional stealth options
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins-discovery")
        chrome_options.add_argument("--disable-web-security")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("Chrome WebDriver initialized successfully")
            return True
        except Exception as e:
            print(f"Error setting up WebDriver: {e}")
            print("Make sure ChromeDriver is installed and in PATH")
            return False
    
    def extract_video_id(self, url):
        """
        Extract video ID from YouTube URL
        # Parses both youtube.com/watch?v= and youtu.be/ formats using regex
        # Returns the 11-character video ID or None if not found
        """
        pattern = r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)'
        match = re.search(pattern, url)
        return match.group(1) if match else None
    
    def scroll_to_load_comments(self, target_comments=100):
        """
        Scroll down the page to load more comments
        # YouTube loads comments dynamically as user scrolls
        # Simulates human scrolling behavior with random delays
        # Stops when target comment count is reached or no new comments load
        """
        print("Scrolling to load comments...")
        last_height = self.driver.execute_script("return document.documentElement.scrollHeight")
        loaded_comments = 0
        
        while loaded_comments < target_comments:
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            
            # Wait for new comments to load
            time.sleep(random.uniform(2, 4))  # Random delay to appear human
            
            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.documentElement.scrollHeight")
            
            # Count currently loaded comments
            try:
                comment_elements = self.driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
                loaded_comments = len(comment_elements)
                print(f"Loaded {loaded_comments} comment threads...")
                
                if loaded_comments >= target_comments:
                    print(f"Target of {target_comments} comments reached!")
                    break
                    
            except Exception as e:
                print(f"Error counting comments: {e}")
            
            # Break if no new content loaded
            if new_height == last_height:
                print("No more comments to load")
                break
            last_height = new_height
            
            # Safety break after too many scrolls
            if loaded_comments == 0 and new_height > 50000:  # Arbitrary large number
                print("Scrolled too far without finding comments")
                break
    
    def extract_comment_data(self, comment_element):
        """
        Extract data from a single comment element
        # Parses individual comment DOM structure for text, author, and metadata
        # Handles missing elements gracefully with try-catch blocks
        # Returns structured comment dictionary or None if extraction fails
        """
        try:
            # Extract comment text
            text_element = comment_element.find_element(By.CSS_SELECTOR, "#content-text")
            comment_text = text_element.text.strip()
            
            # Extract author name
            try:
                author_element = comment_element.find_element(By.CSS_SELECTOR, "#author-text span")
                author_name = author_element.text.strip()
            except:
                author_name = "Unknown User"
            
            # Extract timestamp
            try:
                time_element = comment_element.find_element(By.CSS_SELECTOR, ".published-time-text a")
                timestamp = time_element.text.strip()
            except:
                timestamp = "Unknown time"
            
            # Extract like count
            try:
                like_element = comment_element.find_element(By.CSS_SELECTOR, "#vote-count-middle")
                like_text = like_element.text.strip()
                likes = int(like_text) if like_text.isdigit() else 0
            except:
                likes = 0
            
            # Only return if we have meaningful text
            if len(comment_text) > 5:
                return {
                    'comment_id': f"comment_{hash(comment_text + author_name)}",
                    'text': comment_text,
                    'author': author_name,
                    'timestamp': timestamp,
                    'likes': likes,
                    'video_id': self.extract_video_id(self.driver.current_url)
                }
            
        except Exception as e:
            print(f"Error extracting comment data: {e}")
        
        return None
    
    def scrape_comments_selenium(self, video_url, max_comments=100):
        """
        Main selenium-based comment scraping method
        # Uses Chrome WebDriver to load YouTube page and extract comments
        # Handles dynamic loading and JavaScript-rendered content
        # Returns list of structured comment dictionaries
        """
        print(f"Starting Selenium scraping for: {video_url}")
        
        if not self.setup_driver():
            return []
        
        try:
            # Navigate to video page
            print("Loading YouTube video page...")
            self.driver.get(video_url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "ytd-app"))
            )
            
            # Scroll down to comments section
            print("Scrolling to comments section...")
            self.driver.execute_script("window.scrollTo(0, 1000);")
            time.sleep(3)
            
            # Wait for comments to start loading
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-comments"))
                )
                print("Comments section found!")
            except TimeoutException:
                print("Comments section not found - video might have comments disabled")
                return []
            
            # Scroll to load more comments
            self.scroll_to_load_comments(max_comments)
            
            # Extract all comment elements
            print("Extracting comment data...")
            comment_elements = self.driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
            
            comments = []
            for i, element in enumerate(comment_elements[:max_comments]):
                comment_data = self.extract_comment_data(element)
                if comment_data:
                    comments.append(comment_data)
                    if len(comments) % 10 == 0:
                        print(f"Extracted {len(comments)} comments...")
            
            print(f"Successfully extracted {len(comments)} comments")
            return comments
            
        except Exception as e:
            print(f"Error during Selenium scraping: {e}")
            return []
        
        finally:
            if self.driver:
                self.driver.quit()
                print("WebDriver closed")
    
    def scrape_comments_requests(self, video_url, max_comments=100):
        """
        Fallback method using requests and BeautifulSoup
        # Attempts basic HTML scraping without JavaScript execution
        # Limited effectiveness due to YouTube's dynamic loading
        # Used as backup when Selenium fails
        """
        print("Attempting basic HTML scraping (limited effectiveness)...")
        
        try:
            headers = {
                'User-Agent': random.choice(self.chrome_user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(video_url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for any text that might be comments
            # This is very limited as YouTube loads comments via JavaScript
            potential_comments = []
            
            # Search in script tags for any comment-like JSON data
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'commentRenderer' in script.string:
                    print("Found potential comment data in script tags")
                    # This would require complex JSON parsing
                    break
            
            print("Basic HTML scraping completed (likely no comments found)")
            return potential_comments
            
        except Exception as e:
            print(f"Error in basic HTML scraping: {e}")
            return []
    
    def scrape_comments(self, video_url, max_comments=100, use_selenium=True):
        """
        Main method to scrape comments with multiple strategies
        # Primary entry point that tries different scraping approaches
        # Selenium first for best results, falls back to requests if needed
        # Returns list of comment dictionaries or empty list if all methods fail
        """
        print(f"Scraping comments from: {video_url}")
        
        if use_selenium:
            print("Using Selenium WebDriver for dynamic content...")
            self.comments = self.scrape_comments_selenium(video_url, max_comments)
            
            if len(self.comments) >= 10:
                print(f"Selenium scraping successful: {len(self.comments)} comments")
                return self.comments
            else:
                print("Selenium scraping yielded few comments, trying fallback...")
        
        # Fallback to basic requests
        print("Trying basic HTML scraping as fallback...")
        fallback_comments = self.scrape_comments_requests(video_url, max_comments)
        
        if fallback_comments:
            self.comments = fallback_comments
            return self.comments
        
        print("All scraping methods failed. Possible reasons:")
        print("1. Comments are disabled for this video")
        print("2. Video is private or age-restricted")
        print("3. YouTube's anti-bot measures are blocking requests")
        print("4. ChromeDriver is not properly installed")
        print("\nConsider using YouTube Data API for reliable access")
        
        return []
    
    def save_to_csv(self, filename='raw_comments.csv'):
        """
        Save comments to CSV file
        # Writes collected comments to CSV format with proper headers
        # Handles encoding for special characters and emojis
        # Creates structured data file for next pipeline steps
        """
        if not self.comments:
            print("No comments to save")
            return
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['comment_id', 'text', 'author', 'timestamp', 'likes', 'video_id']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            writer.writeheader()
            for comment in self.comments:
                writer.writerow(comment)
        
        print(f"Comments saved to {filename}")

# Usage example
if __name__ == "__main__":
    scraper = YouTubeScraper()
    
    # Your YouTube URL
    video_url = "https://www.youtube.com/watch?v=Ez8F0nW6S-w"
    
    # Scrape comments with Selenium (real scraping)
    comments = scraper.scrape_comments(video_url, max_comments=100, use_selenium=True)
    
    if comments:
        # Save to CSV
        scraper.save_to_csv('data/raw_comments.csv')
        print(f"Step 1 Complete: {len(comments)} real comments scraped and saved!")
    else:
        print("Step 1 Failed: No comments could be scraped")
        print("Please check the troubleshooting steps above")
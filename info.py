'''First intall the neccesary libraries, or you can opt to use upon use
    the first one is an example'''

pip install python-dotenv openai requests logging time random


import openai
import requests
import json
import os
from typing import Dict, List, Tuple
import logging
import time
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LearningNavigator:
    """
    AI Learning Navigator that provides explanations and curated resources 
    (video, article, social media post) for any question.
    """
    
    def __init__(self):
        """Initialize the Learning Navigator with API keys and configurations."""
        # API Keys (loaded from environment variables for security)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

        # Initialize the OpenAI client
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Verify API keys are available
        self._verify_api_keys()

    def _verify_api_keys(self):
        """Verify that all necessary API keys are available."""
        missing_keys = []
        
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.youtube_api_key:
            missing_keys.append("YOUTUBE_API_KEY")
        if not self.news_api_key:
            missing_keys.append("NEWS_API_KEY")
        if not self.twitter_bearer_token:
            missing_keys.append("TWITTER_BEARER_TOKEN")
            
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
            logger.info("Some features may be limited or unavailable.")

    def get_explanation(self, query: str) -> str:
        """
        Generate a concise explanation for the query using OpenAI's API.
        
        Args:
            query: The user's question
            
        Returns:
            A concise explanation of the topic
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an educational AI that provides clear, concise explanations. "
                                                 "Keep your response between 3-5 sentences, focusing on the core concepts."},
                    {"role": "user", "content": query}
                ],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting explanation from OpenAI: {e}")
            return "I couldn't generate an explanation at the moment. Please try again later."

    def search_youtube_video(self, query: str) -> Dict:
        """
        Search for an educational YouTube video related to the query.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary with video title, URL, and channel name
        """
        if not self.youtube_api_key:
            logger.warning("YouTube API key not available. Using mock data.")
            return self._get_mock_youtube_result(query)
            
        try:
            search_query = f"educational {query} explanation"
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": search_query,
                "type": "video",
                "maxResults": 5,
                "relevanceLanguage": "en",
                "videoEmbeddable": "true",
                "key": self.youtube_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            search_results = response.json()
            
            if not search_results.get("items"):
                return self._get_mock_youtube_result(query)
                
            # Get the first result
            video = search_results["items"][0]
            video_id = video["id"]["videoId"]
            video_title = video["snippet"]["title"]
            channel_name = video["snippet"]["channelTitle"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            return {
                "title": video_title,
                "url": video_url,
                "channel": channel_name
            }
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
            return self._get_mock_youtube_result(query)

    def _get_mock_youtube_result(self, query: str) -> Dict:
        """Generate a mock YouTube result when API is unavailable."""
        return {
            "title": f"Educational video about {query}",
            "url": "https://www.youtube.com/results?search_query=" + query.replace(" ", "+"),
            "channel": "Educational Channel"
        }

    def search_news_article(self, query: str) -> Dict:
        """
        Search for a relevant news article or educational resource related to the query.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary with article title, URL, and source
        """
        if not self.news_api_key:
            logger.warning("News API key not available. Using mock data.")
            return self._get_mock_article_result(query)
            
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "sortBy": "relevancy",
                "language": "en",
                "pageSize": 5,
                "apiKey": self.news_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            search_results = response.json()
            
            if not search_results.get("articles"):
                return self._get_mock_article_result(query)
                
            # Get the first result
            article = search_results["articles"][0]
            
            return {
                "title": article["title"],
                "url": article["url"],
                "source": article["source"]["name"]
            }
        except Exception as e:
            logger.error(f"Error searching for articles: {e}")
            return self._get_mock_article_result(query)

    def _get_mock_article_result(self, query: str) -> Dict:
        """Generate a mock article result when API is unavailable."""
        domains = ["medium.com", "wikipedia.org", "khanacademy.org", "britannica.com"]
        return {
            "title": f"Understanding {query}: A Comprehensive Guide",
            "url": f"https://{random.choice(domains)}/search?q={query.replace(' ', '+')}",
            "source": "Educational Resource"
        }

    def search_twitter_post(self, query: str) -> Dict:
        """
        Search for a relevant Twitter/X post related to the query.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary with tweet text, URL, and author
        """
        if not self.twitter_bearer_token:
            logger.warning("Twitter Bearer Token not available. Using mock data.")
            return self._get_mock_twitter_result(query)
            
        try:
            # Search endpoint for Twitter API v2
            url = "https://api.twitter.com/2/tweets/search/recent"
            
            # Add educational terms to improve search quality
            search_query = f"{query} -is:retweet has:links lang:en"
            
            params = {
                "query": search_query,
                "max_results": 10,
                "tweet.fields": "created_at,author_id,public_metrics",
                "expansions": "author_id",
                "user.fields": "name,username"
            }
            
            headers = {
                "Authorization": f"Bearer {self.twitter_bearer_token}"
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            search_results = response.json()
            
            if not search_results.get("data"):
                return self._get_mock_twitter_result(query)
                
            # Find tweet with highest engagement
            tweets = search_results["data"]
            users = {user["id"]: user for user in search_results.get("includes", {}).get("users", [])}
            
            # Sort by engagement (likes + retweets)
            tweets.sort(key=lambda x: x["public_metrics"]["like_count"] + x["public_metrics"]["retweet_count"], reverse=True)
            
            # Get the best result
            best_tweet = tweets[0]
            tweet_id = best_tweet["id"]
            tweet_text = best_tweet["text"]
            author_id = best_tweet["author_id"]
            author_username = users.get(author_id, {}).get("username", "user")
            author_name = users.get(author_id, {}).get("name", "Twitter User")
            tweet_url = f"https://twitter.com/{author_username}/status/{tweet_id}"
            
            return {
                "text": tweet_text[:100] + "..." if len(tweet_text) > 100 else tweet_text,
                "url": tweet_url,
                "author": author_name
            }
        except Exception as e:
            logger.error(f"Error searching Twitter: {e}")
            return self._get_mock_twitter_result(query)

    def _get_mock_twitter_result(self, query: str) -> Dict:
        """Generate a mock Twitter result when API is unavailable."""
        return {
            "text": f"Check out this interesting information about {query} I just learned #learning #education",
            "url": f"https://twitter.com/search?q={query.replace(' ', '%20')}",
            "author": "Education Expert"
        }

    def generate_response(self, query: str) -> Dict:
        """
        Generate a complete response with explanation and resources.
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary with explanation and resources
        """
        logger.info(f"Processing query: {query}")
        
        # Get the explanation
        explanation = self.get_explanation(query)
        
        # Get resources in parallel (in a real-world application)
        # For simplicity, we're calling them sequentially here
        video = self.search_youtube_video(query)
        article = self.search_news_article(query)
        tweet = self.search_twitter_post(query)
        
        return {
            "query": query,
            "explanation": explanation,
            "resources": {
                "video": video,
                "article": article,
                "tweet": tweet
            }
        }

    def format_response(self, response_data: Dict) -> str:
        """
        Format the response into a readable text format.
        
        Args:
            response_data: The response data generated by generate_response
            
        Returns:
            A formatted string response
        """
        query = response_data["query"]
        explanation = response_data["explanation"]
        video = response_data["resources"]["video"]
        article = response_data["resources"]["article"]
        tweet = response_data["resources"]["tweet"]
        
        formatted_response = f"""
📚 AI LEARNING NAVIGATOR: {query}

{explanation}

📹 VIDEO RESOURCE:
"{video['title']}" by {video['channel']}
🔗 {video['url']}

📄 ARTICLE RESOURCE:
"{article['title']}" from {article['source']}
🔗 {article['url']}

🐦 X/TWITTER DISCUSSION:
@{tweet['author']}: "{tweet['text']}"
🔗 {tweet['url']}
"""
        return formatted_response.strip()

    def answer(self, query: str) -> str:
        """
        Process a user question and return a formatted response.
        
        Args:
            query: The user's question
            
        Returns:
            A formatted response with explanation and resources
        """
        response_data = self.generate_response(query)
        return self.format_response(response_data)


def main():
    """Main function to run the Learning Navigator as a CLI tool."""
    print("🤖 AI Learning Navigator")
    print("Ask me anything and I'll provide an explanation plus resources to learn more.")
    print("Type 'exit' to quit.\n")
    
    navigator = LearningNavigator()
    
    while True:
        query = input("\n❓ What would you like to learn about? ")
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Thanks for learning with AI Learning Navigator! Goodbye!")
            break
            
        if not query.strip():
            continue
            
        print("\nThinking...\n")
        
        try:
            response = navigator.answer(query)
            print(response)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print("Sorry, I encountered an error while processing your request. Please try again.")


if __name__ == "__main__":
    main()
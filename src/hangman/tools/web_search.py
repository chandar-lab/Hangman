import os
from abc import ABC, abstractmethod
from typing import List, Dict

from langchain_core.tools import tool
from dotenv import load_dotenv

from ddgs import DDGS
from tavily import TavilyClient

load_dotenv('.env.local')

# --- Strategy Pattern: Define a common interface for all search providers ---

class SearchProvider(ABC):
    """Abstract Base Class for a search provider."""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> str:
        """
        Performs a search and returns a formatted string of the results.
        
        Args:
            query: The search query string.
            max_results: The maximum number of results to return.
            
        Returns:
            A formatted string containing the search results.
        """
        pass

# --- Concrete Implementation for Tavily ---

class TavilySearchProvider(SearchProvider):
    """Search provider implementation using the Tavily API."""
    
    def __init__(self):
        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY not found in environment variables.")
            
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def search(self, query: str, max_results: int = 5) -> str:
        print(f"---TOOL: TavilySearch--- Query: {query}")
        try:
            # Tavily's search is optimized for LLMs and returns concise results.
            response = self.client.search(query=query, search_depth="basic", max_results=max_results)
            results = response.get('results', [])
            
            if not results:
                return "No results found."

            formatted_results = []
            for i, res in enumerate(results):
                formatted_results.append(
                    f"Source {i+1}: {res.get('title', 'N/A')}\n"
                    f"URL: {res.get('url', 'N/A')}\n"
                    f"Snippet: {res.get('content', 'N/A')}"
                )
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"An error occurred during Tavily search: {e}"

# --- Concrete Implementation for DuckDuckGo ---

class DuckDuckGoSearchProvider(SearchProvider):
    """Search provider implementation using the duckduckgo-search library."""

    def __init__(self):
        self.client = DDGS()

    def search(self, query: str, max_results: int = 5) -> str:
        print(f"---TOOL: DuckDuckGoSearch--- Query: {query}")
        try:
            # DDGS returns a list of dictionaries.
            results = self.client.text(query, max_results=max_results)
            
            if not results:
                return f"No results found. - Results: {results}"

            formatted_results = []
            for i, res in enumerate(results):
                formatted_results.append(
                    f"Source {i+1}: {res.get('title', 'N/A')}\n"
                    f"URL: {res.get('href', 'N/A')}\n"
                    f"Snippet: {res.get('body', 'N/A')}"
                )
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"An error occurred during DuckDuckGo search: {e}"


# --- Factory Function to Create and Configure the Tool ---

def get_search_tool(provider: str) -> tool:
    """
    Factory function that creates a web search tool configured with a specific provider.
    
    Args:
        provider (str): The name of the provider to use. 
                        Supported: 'tavily', 'duckduckgo'.
    
    Returns:
        A configured LangChain tool instance for web search.
    """
    provider_name = provider.lower()
    
    if provider_name == "tavily":
        search_provider_instance = TavilySearchProvider()
    elif provider_name == "duckduckgo":
        search_provider_instance = DuckDuckGoSearchProvider()
    else:
        raise ValueError(f"Unknown search provider: '{provider_name}'. Supported providers are 'tavily' and 'duckduckgo'.")

    @tool
    def web_search(query: str) -> str:
        """
        Searches the internet for up-to-date information on a given topic.
        Use this for questions about recent events, current affairs, or facts
        not contained in your internal knowledge base.
        """
        return search_provider_instance.search(query)

    return web_search
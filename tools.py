import arxiv
from smolagents import tool
from datetime import datetime, timezone
import random
@tool
def search_arxiv(query: str, max_results: int = 3) -> list:
    """
    Searches arXiv for academic papers based on a query.
    
    Args:
        query: The search terms (e.g., 'Neural SDEs time series').
        max_results: The maximum number of papers to return. Defaults to 3.
        
    Returns:
        A list of dictionaries. Each dict contains 'title', 'authors', 'year', 'url', and 'abstract'.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, 
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance # Ensures accurate semantic matching
    )
    
    results = []
    for paper in client.results(search):
        # Extract authors into a comma-separated string
        authors = ", ".join([author.name for author in paper.authors])
        
        # Append as a structured dictionary rather than a raw string
        results.append({
            "title": paper.title,
            "authors": authors,
            "year": paper.published.year,
            "url": paper.entry_id,
            "abstract": paper.summary
        })
        
    return results

@tool
def get_inspirational_quote() -> str:
    """Returns a random inspirational quote to inspire the day's reflection."""
    quotes = [
    "Start where you are, use what you have, do what you can.",
    "Small steps every day add up to big changes over time.",
    "You don't have to be perfect to be great.",
    "The best time to begin was yesterday. The next best time is now.",
    "Difficult roads often lead to beautiful destinations.",
    "Your only limit is the one you set for yourself.",
    "Every expert was once a beginner.",
    "Progress, not perfection, is the goal.",
    "Courage is taking the next step even when you're afraid.",
    "Believe you can, and you're halfway there.",
    "Failure is just practice for success.",
    "Dream big, start small, and never stop moving forward.",
    "The harder you work for something, the more it means when you get it.",
    "Be patient with yourself. Growth takes time.",
    "Great things are built one day at a time.",
    "Don't wait for motivation. Build the habit and the motivation will follow.",
    "A setback is a setup for a comeback.",
    "You are stronger than you think and braver than you feel.",
    "Focus on the step in front of you, not the whole staircase.",
    "Today's effort is tomorrow's strength."
]
    return random.choice(quotes)

@tool
def get_current_time() -> str:
    """
    Returns the current date and time in UTC.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

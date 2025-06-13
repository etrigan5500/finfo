# Google Custom Search JSON API Setup Guide

This guide will help you set up Google Custom Search JSON API for the stock prediction application.

## Benefits Over SerpAPI

- **100 FREE requests per day** (vs SerpAPI's 100/month)
- **Official Google API** with better reliability  
- **No subscription required** after free quota
- **Same quality news results**

## Step 1: Get Google Custom Search JSON API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the "Custom Search API":
   - Go to "APIs & Services" → "Library"
   - Search for "Custom Search API"
   - Click "Enable"
4. Create credentials:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy your API key and save it securely

## Step 2: Create Custom Search Engine

1. Go to [Google Custom Search](https://cse.google.com/cse/)
2. Click "Add" to create a new search engine
3. **Setup Configuration:**
   - **Sites to search**: You can either:
     - Enter `*` to search the entire web
     - Or add specific news sites: `reuters.com`, `bloomberg.com`, `cnbc.com`
   - **Name**: "Stock News Search" (or any name you prefer)
   - **Language**: English
4. Click "Create"
5. **Get your Search Engine ID:**
   - Click on your newly created search engine
   - Go to "Setup" tab
   - Copy the **Search engine ID** (starts with a letter, looks like: `017576662512468239146:omuauf_lfve`)

## Step 3: Configure Environment Variables

Create or update your `.env` file:

```bash
# Google Custom Search API Configuration
GOOGLE_API_KEY=your_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

# Other APIs (optional)
GEMINI_API_KEY=your_gemini_api_key_here
```

## Step 4: Test Your Setup

Run this Python code to test:

```python
import requests

api_key = "your_api_key"
search_engine_id = "your_cx_id"
query = "Apple financial news"

url = "https://www.googleapis.com/customsearch/v1"
params = {
    "key": api_key,
    "cx": search_engine_id,
    "q": query,
    "num": 5
}

response = requests.get(url, params=params)
if response.status_code == 200:
    results = response.json()
    print(f"Found {len(results.get('items', []))} results")
    for item in results.get('items', [])[:3]:
        print(f"- {item['title']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Usage Quotas and Limits

- **Free Quota**: 100 searches per day
- **Rate Limit**: 10 queries per second
- **Results per query**: Maximum 10 items per request
- **For training**: 40 queries per run (20 companies × 2 queries each)
- **Daily capacity**: ~2.5 training runs per day

## Cost Comparison

| API | Free Tier | Cost After | Training Runs/Month |
|-----|-----------|------------|-------------------|
| Google Custom Search | 100/day | $5 per 1000 queries | ~75 runs |
| SerpAPI | 100/month | $50/month | ~2.5 runs |

## Troubleshooting

### Error: "API key not valid"
- Check that Custom Search API is enabled in Google Cloud Console
- Verify API key is correct and not restricted

### Error: "Invalid cx parameter"
- Ensure Search Engine ID (cx) is correct
- Make sure the custom search engine is active

### Error: "Quota exceeded"
- You've reached the 100 daily requests limit
- Wait until the next day or upgrade to paid plan

### No results returned
- Try broader search terms
- Check if your custom search engine is set to search entire web (`*`)
- Ensure news sites are included in your search scope

## Advanced Configuration

### Restrict to News Sites Only

When creating your Custom Search Engine, add these sites specifically:

```
reuters.com
bloomberg.com
cnbc.com
marketwatch.com
finance.yahoo.com
google.com/finance
cnn.com/business
wsj.com
ft.com
```

### API Parameters for News

The application uses these parameters for better news results:

```python
params = {
    "key": api_key,
    "cx": search_engine_id, 
    "q": f"{query} site:reuters.com OR site:bloomberg.com",
    "num": 10,
    "dateRestrict": "m1",  # Last month only
    "safe": "active",
    "fields": "items(title,snippet,link,displayLink)"
}
```

## Integration with Training

Once set up, the training notebook will use:
- **2 queries per company**: financial news + stock prediction news
- **10 results per query** = 20 articles per company
- **Clustering**: Groups articles into 3 topic clusters
- **Final selection**: 9 representative articles per company

This provides diverse, recent news data for training the sentiment analysis model. 
# Crawl4AI Custom LLM Context
Generated on: 2025-08-31T11:25:17.960Z
Total files: 2
Estimated tokens: 17,663

---

## Configuration Objects - Full Content
Component ID: config_objects
Context Type: memory
Estimated tokens: 7,868

## Browser, Crawler & LLM Configuration

Core configuration classes for controlling browser behavior, crawl operations, LLM providers, and understanding crawl results.

### BrowserConfig - Browser Environment Setup

```python
from crawl4ai import BrowserConfig, AsyncWebCrawler

# Basic browser configuration
browser_config = BrowserConfig(
    browser_type="chromium",  # "chromium", "firefox", "webkit"
    headless=True,           # False for visible browser (debugging)
    viewport_width=1280,
    viewport_height=720,
    verbose=True
)

# Advanced browser setup with proxy and persistence
browser_config = BrowserConfig(
    headless=False,
    proxy="http://username:password@proxy.example.com:8080",  # Use environment variables in production
    use_persistent_context=True,
    user_data_dir="./browser_data",
    cookies=[
        {"name": "session", "value": "abc123", "domain": "example.com"}
    ],
    headers={"Accept-Language": "en-US,en;q=0.9"},
    user_agent="Mozilla/5.0 (X11; Linux x86_64) Chrome/116.0.0.0 Safari/537.36",
    text_mode=True,  # Disable images for faster crawling
    extra_args=["--disable-extensions", "--no-sandbox"]
)

async with AsyncWebCrawler(config=browser_config) as crawler:
    result = await crawler.arun("https://example.com")
```

### CrawlerRunConfig - Crawl Operation Control

```python
from crawl4ai import CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

# Basic crawl configuration
run_config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    word_count_threshold=10,
    excluded_tags=["nav", "footer", "script"],
    exclude_external_links=True,
    screenshot=True,
    pdf=True
)

# Advanced content processing
md_generator = DefaultMarkdownGenerator(
    content_filter=PruningContentFilter(threshold=0.6),
    options={"citations": True, "ignore_links": False}
)

run_config = CrawlerRunConfig(
    # Content processing
    markdown_generator=md_generator,
    css_selector="main.content",  # Focus on specific content
    target_elements=[".article", ".post"],  # Multiple target selectors
    process_iframes=True,
    remove_overlay_elements=True,

    # Page interaction
    js_code=[
        "window.scrollTo(0, document.body.scrollHeight);",
        "document.querySelector('.load-more')?.click();"
    ],
    wait_for="css:.content-loaded",
    wait_for_timeout=10000,
    scan_full_page=True,

    # Session management
    session_id="persistent_session",

    # Media handling
    screenshot=True,
    pdf=True,
    capture_mhtml=True,
    image_score_threshold=5,

    # Advanced options
    simulate_user=True,
    magic=True,  # Auto-handle popups
    verbose=True
)
```

### CrawlerRunConfig Parameters by Category

```python
# Content Processing
config = CrawlerRunConfig(
    word_count_threshold=10,              # Min words per content block
    css_selector="main.article",          # Focus on specific content
    target_elements=[".post", ".content"], # Multiple target selectors
    excluded_tags=["nav", "footer"],       # Remove these tags
    excluded_selector="#ads, .tracker",   # Remove by selector
    only_text=True,                       # Text-only extraction
    keep_data_attributes=True,            # Preserve data-* attributes
    remove_forms=True,                    # Remove all forms
    process_iframes=True                  # Include iframe content
)

# Page Navigation & Timing
config = CrawlerRunConfig(
    wait_until="networkidle",             # Wait condition
    page_timeout=60000,                   # 60 second timeout
    wait_for="css:.loaded",               # Wait for specific element
    wait_for_images=True,                 # Wait for images to load
    delay_before_return_html=0.5,         # Final delay before capture
    semaphore_count=10                    # Max concurrent operations
)

# Page Interaction
config = CrawlerRunConfig(
    js_code="document.querySelector('button').click();",
    scan_full_page=True,                  # Auto-scroll page
    scroll_delay=0.3,                     # Delay between scrolls
    remove_overlay_elements=True,         # Remove popups/modals
    simulate_user=True,                   # Simulate human behavior
    override_navigator=True,              # Override navigator properties
    magic=True                           # Auto-handle common patterns
)

# Caching & Session
config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,          # Cache behavior
    session_id="my_session",              # Persistent session
    shared_data={"context": "value"}      # Share data between hooks
)

# Media & Output
config = CrawlerRunConfig(
    screenshot=True,                      # Capture screenshot
    pdf=True,                            # Generate PDF
    capture_mhtml=True,                  # Capture MHTML archive
    image_score_threshold=3,             # Filter low-quality images
    exclude_external_images=True         # Remove external images
)

# Link & Domain Filtering
config = CrawlerRunConfig(
    exclude_external_links=True,         # Remove external links
    exclude_social_media_links=True,     # Remove social media links
    exclude_domains=["ads.com", "tracker.io"],  # Custom domain filter
    exclude_internal_links=False         # Keep internal links
)
```

### LLMConfig - Language Model Setup

```python
from crawl4ai import LLMConfig

# OpenAI configuration
llm_config = LLMConfig(
    provider="openai/gpt-4o-mini",
    api_token=os.getenv("OPENAI_API_KEY"),  # or "env:OPENAI_API_KEY"
    temperature=0.1,
    max_tokens=2000
)

# Local model with Ollama
llm_config = LLMConfig(
    provider="ollama/llama3.3",
    api_token=None,  # Not needed for Ollama
    base_url="http://localhost:11434"  # Custom endpoint
)

# Anthropic Claude
llm_config = LLMConfig(
    provider="anthropic/claude-3-5-sonnet-20240620",
    api_token="env:ANTHROPIC_API_KEY",
    max_tokens=4000
)

# Google Gemini
llm_config = LLMConfig(
    provider="gemini/gemini-1.5-pro",
    api_token="env:GEMINI_API_KEY"
)

# Groq (fast inference)
llm_config = LLMConfig(
    provider="groq/llama3-70b-8192",
    api_token="env:GROQ_API_KEY"
)
```

### CrawlResult - Understanding Output

```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

async with AsyncWebCrawler() as crawler:
    result = await crawler.arun("https://example.com", config=run_config)

    # Basic status information
    print(f"Success: {result.success}")
    print(f"Status: {result.status_code}")
    print(f"URL: {result.url}")

    if not result.success:
        print(f"Error: {result.error_message}")
        return

    # HTML content variants
    print(f"Original HTML: {len(result.html)} chars")
    print(f"Cleaned HTML: {len(result.cleaned_html or '')} chars")

    # Markdown output (MarkdownGenerationResult)
    if result.markdown:
        print(f"Raw markdown: {len(result.markdown.raw_markdown)} chars")
        print(f"With citations: {len(result.markdown.markdown_with_citations)} chars")

        # Filtered content (if content filter was used)
        if result.markdown.fit_markdown:
            print(f"Fit markdown: {len(result.markdown.fit_markdown)} chars")
            print(f"Fit HTML: {len(result.markdown.fit_html)} chars")

    # Extracted structured data
    if result.extracted_content:
        import json
        data = json.loads(result.extracted_content)
        print(f"Extracted {len(data)} items")

    # Media and links
    images = result.media.get("images", [])
    print(f"Found {len(images)} images")
    for img in images[:3]:  # First 3 images
        print(f"  {img.get('src')} (score: {img.get('score', 0)})")

    internal_links = result.links.get("internal", [])
    external_links = result.links.get("external", [])
    print(f"Links: {len(internal_links)} internal, {len(external_links)} external")

    # Generated files
    if result.screenshot:
        print(f"Screenshot captured: {len(result.screenshot)} chars (base64)")
        # Save screenshot
        import base64
        with open("page.png", "wb") as f:
            f.write(base64.b64decode(result.screenshot))

    if result.pdf:
        print(f"PDF generated: {len(result.pdf)} bytes")
        with open("page.pdf", "wb") as f:
            f.write(result.pdf)

    if result.mhtml:
        print(f"MHTML captured: {len(result.mhtml)} chars")
        with open("page.mhtml", "w", encoding="utf-8") as f:
            f.write(result.mhtml)

    # SSL certificate information
    if result.ssl_certificate:
        print(f"SSL Issuer: {result.ssl_certificate.issuer}")
        print(f"Valid until: {result.ssl_certificate.valid_until}")

    # Network and console data (if captured)
    if result.network_requests:
        requests = [r for r in result.network_requests if r.get("event_type") == "request"]
        print(f"Network requests captured: {len(requests)}")

    if result.console_messages:
        errors = [m for m in result.console_messages if m.get("type") == "error"]
        print(f"Console messages: {len(result.console_messages)} ({len(errors)} errors)")

    # Session and metadata
    if result.session_id:
        print(f"Session ID: {result.session_id}")

    if result.metadata:
        print(f"Metadata: {result.metadata.get('title', 'No title')}")
```

### Configuration Helpers and Best Practices

```python
# Clone configurations for variations
base_config = CrawlerRunConfig(
    cache_mode=CacheMode.ENABLED,
    word_count_threshold=200,
    verbose=True
)

# Create streaming version
stream_config = base_config.clone(
    stream=True,
    cache_mode=CacheMode.BYPASS
)

# Create debug version
debug_config = base_config.clone(
    headless=False,
    page_timeout=120000,
    verbose=True
)

# Serialize/deserialize configurations
config_dict = base_config.dump()  # Convert to dict
restored_config = CrawlerRunConfig.load(config_dict)  # Restore from dict

# Browser configuration management
browser_config = BrowserConfig(headless=True, text_mode=True)
browser_dict = browser_config.to_dict()
cloned_browser = browser_config.clone(headless=False, verbose=True)
```

### Common Configuration Patterns

```python
# Fast text-only crawling
fast_config = CrawlerRunConfig(
    cache_mode=CacheMode.ENABLED,
    text_mode=True,
    exclude_external_links=True,
    exclude_external_images=True,
    word_count_threshold=50
)

# Comprehensive data extraction
comprehensive_config = CrawlerRunConfig(
    process_iframes=True,
    scan_full_page=True,
    wait_for_images=True,
    screenshot=True,
    capture_network_requests=True,
    capture_console_messages=True,
    magic=True
)

# Stealth crawling
stealth_config = CrawlerRunConfig(
    simulate_user=True,
    override_navigator=True,
    mean_delay=2.0,
    max_range=1.0,
    user_agent_mode="random"
)
```

### Advanced Configuration Features

#### User Agent Management & Bot Detection Avoidance

```python
from crawl4ai import CrawlerRunConfig

# Random user agent generation
config = CrawlerRunConfig(
    user_agent_mode="random",
    user_agent_generator_config={
        "platform": "windows",  # "windows", "macos", "linux", "android", "ios"
        "browser": "chrome",    # "chrome", "firefox", "safari", "edge"
        "device_type": "desktop"  # "desktop", "mobile", "tablet"
    }
)

# Custom user agent with stealth features
config = CrawlerRunConfig(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    simulate_user=True,      # Simulate human mouse movements
    override_navigator=True,  # Override navigator properties
    mean_delay=1.5,          # Random delays between actions
    max_range=2.0
)

# Combined anti-detection approach
stealth_config = CrawlerRunConfig(
    user_agent_mode="random",
    simulate_user=True,
    override_navigator=True,
    magic=True,  # Auto-handle common bot detection patterns
    delay_before_return_html=2.0
)
```

#### Proxy Configuration with ProxyConfig

```python
import os
from crawl4ai import CrawlerRunConfig, ProxyConfig, ProxyRotationStrategy

# Single proxy configuration
proxy_config = ProxyConfig(
    server="http://proxy.example.com:8080",
    username="proxy_user",
    password=os.getenv("PROXY_PASSWORD")  # Use environment variables
)

# From proxy string format
proxy_config = ProxyConfig.from_string(os.getenv("PROXY_CONFIG"))  # Format: "host:port:username:password"

# Multiple proxies with rotation
proxies = [
    ProxyConfig(server="http://proxy1.example.com:8080", username=os.getenv("PROXY1_USER"), password=os.getenv("PROXY1_PASS")),
    ProxyConfig(server="http://proxy2.example.com:8080", username=os.getenv("PROXY2_USER"), password=os.getenv("PROXY2_PASS")),
    ProxyConfig(server="http://proxy3.example.com:8080", username=os.getenv("PROXY3_USER"), password=os.getenv("PROXY3_PASS"))
]

rotation_strategy = ProxyRotationStrategy(
    proxies=proxies,
    rotation_method="round_robin"  # or "random", "least_used"
)

config = CrawlerRunConfig(
    proxy_config=proxy_config,
    proxy_rotation_strategy=rotation_strategy
)

# Load proxies from environment variable
proxies_from_env = ProxyConfig.from_env("MY_PROXIES")  # comma-separated proxy strings
```

#### Content Selection: css_selector vs target_elements

```python
from crawl4ai import CrawlerRunConfig

# css_selector: Extracts HTML at top level, affects entire processing
config = CrawlerRunConfig(
    css_selector="main.article, .content-area",  # Can be list of selectors
    # Everything else (markdown, extraction, links) works only on this HTML subset
)

# target_elements: Focuses extraction within already processed HTML
config = CrawlerRunConfig(
    css_selector="body",  # First extract entire body
    target_elements=[     # Then focus extraction on these elements
        ".article-content",
        ".post-body",
        ".main-text"
    ],
    # Links, media from entire body, but markdown/extraction only from target_elements
)

# Hierarchical content selection
config = CrawlerRunConfig(
    css_selector=["#main-content", ".article-wrapper"],  # Top-level extraction
    target_elements=[                                     # Subset for processing
        ".article-title",
        ".article-body",
        ".article-metadata"
    ],
    excluded_selector="#sidebar, .ads, .comments"  # Remove these from selection
)
```

#### Advanced wait_for Conditions

```python
from crawl4ai import CrawlerRunConfig

# CSS selector waiting
config = CrawlerRunConfig(
    wait_for="css:.content-loaded",  # Wait for element to appear
    wait_for_timeout=15000
)

# JavaScript boolean expression waiting
config = CrawlerRunConfig(
    wait_for="js:() => window.dataLoaded === true",  # Custom JS condition
    wait_for_timeout=20000
)

# Complex JavaScript conditions
config = CrawlerRunConfig(
    wait_for="js:() => document.querySelectorAll('.item').length >= 10",
    js_code=[
        "document.querySelector('.load-more')?.click();",
        "window.scrollTo(0, document.body.scrollHeight);"
    ]
)

# Multiple conditions with JavaScript
config = CrawlerRunConfig(
    wait_for="js:() => !document.querySelector('.loading') && document.querySelector('.results')",
    page_timeout=30000
)
```

#### Session Management for Multi-Step Crawling

```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

# Persistent session across multiple arun() calls
async def multi_step_crawling():
    async with AsyncWebCrawler() as crawler:
        # Step 1: Login page
        login_config = CrawlerRunConfig(
            session_id="user_session",  # Create persistent session
            js_code="document.querySelector('#username').value = process.env.LOGIN_USER; document.querySelector('#password').value = process.env.LOGIN_PASS; document.querySelector('#login').click();",
            wait_for="css:.dashboard",
            cache_mode=CacheMode.BYPASS
        )

        result1 = await crawler.arun("https://example.com/login", config=login_config)

        # Step 2: Navigate to protected area (reuses same browser page)
        nav_config = CrawlerRunConfig(
            session_id="user_session",  # Same session = same browser page
            js_only=True,  # No page reload, just JS navigation
            js_code="window.location.href = '/dashboard/data';",
            wait_for="css:.data-table"
        )

        result2 = await crawler.arun("https://example.com/dashboard/data", config=nav_config)

        # Step 3: Extract data from multiple pages
        for page in range(1, 6):
            page_config = CrawlerRunConfig(
                session_id="user_session",
                js_only=True,
                js_code=f"document.querySelector('.page-{page}').click();",
                wait_for=f"js:() => document.querySelector('.page-{page}').classList.contains('active')"
            )

            result = await crawler.arun(f"https://example.com/data/page/{page}", config=page_config)
            print(f"Page {page} data extracted: {len(result.extracted_content)}")

        # Important: Kill session when done
        await crawler.kill_session("user_session")

# Session with shared data between steps
async def session_with_shared_data():
    shared_context = {"user_id": "12345", "preferences": {"theme": "dark"}}

    config = CrawlerRunConfig(
        session_id="persistent_session",
        shared_data=shared_context,  # Available across all session calls
        js_code="console.log('User ID:', window.sharedData.user_id);"
    )
```

#### Identity-Based Crawling Parameters

```python
from crawl4ai import CrawlerRunConfig, GeolocationConfig

# Locale and timezone simulation
config = CrawlerRunConfig(
    locale="en-US",                    # Browser language preference
    timezone_id="America/New_York",    # Timezone setting
    user_agent_mode="random",
    user_agent_generator_config={
        "platform": "windows",
        "locale": "en-US"
    }
)

# Geolocation simulation
geo_config = GeolocationConfig(
    latitude=40.7128,   # New York coordinates
    longitude=-74.0060,
    accuracy=100.0
)

config = CrawlerRunConfig(
    geolocation=geo_config,
    locale="en-US",
    timezone_id="America/New_York"
)

# Complete identity simulation
identity_config = CrawlerRunConfig(
    # Location identity
    locale="fr-FR",
    timezone_id="Europe/Paris",
    geolocation=GeolocationConfig(latitude=48.8566, longitude=2.3522),

    # Browser identity
    user_agent_mode="random",
    user_agent_generator_config={
        "platform": "windows",
        "locale": "fr-FR",
        "browser": "chrome"
    },

    # Behavioral identity
    simulate_user=True,
    override_navigator=True,
    mean_delay=2.0,
    max_range=1.5
)
```

#### Simplified Import Pattern

```python
# Almost everything from crawl4ai main package
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    LLMConfig,
    CacheMode,
    ProxyConfig,
    GeolocationConfig
)

import os

# Specialized strategies (still from crawl4ai)
from crawl4ai import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
    DefaultMarkdownGenerator,
    PruningContentFilter,
    RegexChunking
)

# Complete example with simplified imports
async def example_crawl():
    browser_config = BrowserConfig(headless=True)

    run_config = CrawlerRunConfig(
        user_agent_mode="random",
        proxy_config=ProxyConfig.from_string(os.getenv("PROXY_CONFIG")),  # Format: "host:port:username:password"
        css_selector="main.content",
        target_elements=[".article", ".post"],
        wait_for="js:() => document.querySelector('.loaded')",
        session_id="my_session",
        simulate_user=True
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun("https://example.com", config=run_config)
        return result
```

## Advanced Features

Comprehensive guide to advanced crawling capabilities including file handling, authentication, dynamic content, monitoring, and session management.

### File Download Handling

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
import os

# Enable downloads with custom path
downloads_path = os.path.join(os.getcwd(), "my_downloads")
os.makedirs(downloads_path, exist_ok=True)

browser_config = BrowserConfig(
    accept_downloads=True,
    downloads_path=downloads_path
)

# Trigger downloads with JavaScript
async def download_files():
    async with AsyncWebCrawler(config=browser_config) as crawler:
        config = CrawlerRunConfig(
            js_code="""
                // Click download links
                const downloadLinks = document.querySelectorAll('a[href$=".pdf"]');
                for (const link of downloadLinks) {
                    link.click();
                    await new Promise(r => setTimeout(r, 2000));  // Delay between downloads
                }
            """,
            wait_for=5  # Wait for downloads to start
        )

        result = await crawler.arun("https://example.com/downloads", config=config)

        if result.downloaded_files:
            print("Downloaded files:")
            for file_path in result.downloaded_files:
                print(f"- {file_path} ({os.path.getsize(file_path)} bytes)")
```

### Hooks & Authentication

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from playwright.async_api import Page, BrowserContext

async def advanced_crawler_with_hooks():
    browser_config = BrowserConfig(headless=True, verbose=True)
    crawler = AsyncWebCrawler(config=browser_config)

    # Hook functions for different stages
    async def on_browser_created(browser, **kwargs):
        print("[HOOK] Browser created successfully")
        return browser

    async def on_page_context_created(page: Page, context: BrowserContext, **kwargs):
        print("[HOOK] Setting up page & context")

        # Block images for faster crawling
        async def route_filter(route):
            if route.request.resource_type == "image":
                await route.abort()
            else:
                await route.continue_()

        await context.route("**", route_filter)

        # Simulate login if needed
        # await page.goto("https://example.com/login")
        # await page.fill("input[name='username']", "testuser")
        # await page.fill("input[name='password']", "password123")
        # await page.click("button[type='submit']")

        await page.set_viewport_size({"width": 1080, "height": 600})
        return page

    async def before_goto(page: Page, context: BrowserContext, url: str, **kwargs):
        print(f"[HOOK] About to navigate to: {url}")
        await page.set_extra_http_headers({"Custom-Header": "my-value"})
        return page

    async def after_goto(page: Page, context: BrowserContext, url: str, response, **kwargs):
        print(f"[HOOK] Successfully loaded: {url}")
        try:
            await page.wait_for_selector('.content', timeout=1000)
            print("[HOOK] Content found!")
        except:
            print("[HOOK] Content not found, continuing")
        return page

    async def before_retrieve_html(page: Page, context: BrowserContext, **kwargs):
        print("[HOOK] Final actions before HTML retrieval")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        return page

    # Attach hooks
    crawler.crawler_strategy.set_hook("on_browser_created", on_browser_created)
    crawler.crawler_strategy.set_hook("on_page_context_created", on_page_context_created)
    crawler.crawler_strategy.set_hook("before_goto", before_goto)
    crawler.crawler_strategy.set_hook("after_goto", after_goto)
    crawler.crawler_strategy.set_hook("before_retrieve_html", before_retrieve_html)

    await crawler.start()

    config = CrawlerRunConfig()
    result = await crawler.arun("https://example.com", config=config)

    if result.success:
        print(f"Crawled successfully: {len(result.html)} chars")

    await crawler.close()
```

### Lazy Loading & Dynamic Content

```python
# Handle lazy-loaded images and infinite scroll
async def handle_lazy_loading():
    config = CrawlerRunConfig(
        # Wait for images to fully load
        wait_for_images=True,

        # Automatically scroll entire page to trigger lazy loading
        scan_full_page=True,
        scroll_delay=0.5,  # Delay between scroll steps

        # JavaScript for custom lazy loading
        js_code="""
            // Scroll and wait for content to load
            window.scrollTo(0, document.body.scrollHeight);

            // Click "Load More" if available
            const loadMoreBtn = document.querySelector('.load-more');
            if (loadMoreBtn) {
                loadMoreBtn.click();
            }
        """,

        # Wait for specific content to appear
        wait_for="css:.lazy-content:nth-child(20)",  # Wait for 20 items

        # Exclude external images to focus on main content
        exclude_external_images=True
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://example.com/gallery", config=config)

        if result.success:
            images = result.media.get("images", [])
            print(f"Loaded {len(images)} images after lazy loading")
            for img in images[:3]:
                print(f"- {img.get('src')} (score: {img.get('score', 'N/A')})")
```

### Network & Console Monitoring

```python
# Capture all network requests and console messages for debugging
async def monitor_network_and_console():
    config = CrawlerRunConfig(
        capture_network_requests=True,
        capture_console_messages=True
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://example.com", config=config)

        if result.success:
            # Analyze network requests
            if result.network_requests:
                requests = [r for r in result.network_requests if r.get("event_type") == "request"]
                responses = [r for r in result.network_requests if r.get("event_type") == "response"]
                failures = [r for r in result.network_requests if r.get("event_type") == "request_failed"]

                print(f"Network activity: {len(requests)} requests, {len(responses)} responses, {len(failures)} failures")

                # Find API calls
                api_calls = [r for r in requests if "api" in r.get("url", "")]
                print(f"API calls detected: {len(api_calls)}")

                # Show failed requests
                for failure in failures[:3]:
                    print(f"Failed: {failure.get('url')} - {failure.get('failure_text')}")

            # Analyze console messages
            if result.console_messages:
                message_types = {}
                for msg in result.console_messages:
                    msg_type = msg.get("type", "unknown")
                    message_types[msg_type] = message_types.get(msg_type, 0) + 1

                print(f"Console messages: {message_types}")

                # Show errors
                errors = [msg for msg in result.console_messages if msg.get("type") == "error"]
                for error in errors[:2]:
                    print(f"JS Error: {error.get('text', '')[:100]}")
```

### Session Management for Multi-Step Workflows

```python
# Maintain state across multiple requests for complex workflows
async def multi_step_session_workflow():
    session_id = "workflow_session"

    async with AsyncWebCrawler() as crawler:
        # Step 1: Initial page load
        config1 = CrawlerRunConfig(
            session_id=session_id,
            wait_for="css:.content-loaded"
        )

        result1 = await crawler.arun("https://example.com/step1", config=config1)
        print("Step 1 completed")

        # Step 2: Navigate and interact (same browser tab)
        config2 = CrawlerRunConfig(
            session_id=session_id,
            js_only=True,  # Don't reload page, just run JS
            js_code="""
                document.querySelector('#next-button').click();
            """,
            wait_for="css:.step2-content"
        )

        result2 = await crawler.arun("https://example.com/step2", config=config2)
        print("Step 2 completed")

        # Step 3: Form submission
        config3 = CrawlerRunConfig(
            session_id=session_id,
            js_only=True,
            js_code="""
                document.querySelector('#form-field').value = 'test data';
                document.querySelector('#submit-btn').click();
            """,
            wait_for="css:.results"
        )

        result3 = await crawler.arun("https://example.com/submit", config=config3)
        print("Step 3 completed")

        # Clean up session
        await crawler.crawler_strategy.kill_session(session_id)

# Advanced GitHub commits pagination example
async def github_commits_pagination():
    session_id = "github_session"
    all_commits = []

    async with AsyncWebCrawler() as crawler:
        for page in range(3):
            if page == 0:
                # Initial load
                config = CrawlerRunConfig(
                    session_id=session_id,
                    wait_for="js:() => document.querySelectorAll('li.Box-sc-g0xbh4-0').length > 0"
                )
            else:
                # Navigate to next page
                config = CrawlerRunConfig(
                    session_id=session_id,
                    js_only=True,
                    js_code='document.querySelector(\'a[data-testid="pagination-next-button"]\').click();',
                    wait_for="js:() => document.querySelectorAll('li.Box-sc-g0xbh4-0').length > 0"
                )

            result = await crawler.arun(
                "https://github.com/microsoft/TypeScript/commits/main",
                config=config
            )

            if result.success:
                commit_count = result.cleaned_html.count('li.Box-sc-g0xbh4-0')
                print(f"Page {page + 1}: Found {commit_count} commits")

        await crawler.crawler_strategy.kill_session(session_id)
```

### SSL Certificate Analysis

```python
# Fetch and analyze SSL certificates
async def analyze_ssl_certificates():
    config = CrawlerRunConfig(
        fetch_ssl_certificate=True
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://example.com", config=config)

        if result.success and result.ssl_certificate:
            cert = result.ssl_certificate

            # Basic certificate info
            print(f"Issuer: {cert.issuer.get('CN', 'Unknown')}")
            print(f"Subject: {cert.subject.get('CN', 'Unknown')}")
            print(f"Valid from: {cert.valid_from}")
            print(f"Valid until: {cert.valid_until}")
            print(f"Fingerprint: {cert.fingerprint}")

            # Export certificate in different formats
            import os
            os.makedirs("certificates", exist_ok=True)

            cert.to_json("certificates/cert.json")
            cert.to_pem("certificates/cert.pem")
            cert.to_der("certificates/cert.der")

            print("Certificate exported in multiple formats")
```

### Advanced Page Interaction

```python
# Complex page interactions with dynamic content
async def advanced_page_interaction():
    async with AsyncWebCrawler() as crawler:
        # Multi-step interaction with waiting
        config = CrawlerRunConfig(
            js_code=[
                # Step 1: Scroll to load content
                "window.scrollTo(0, document.body.scrollHeight);",

                # Step 2: Wait and click load more
                """
                (async () => {
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    const loadMore = document.querySelector('.load-more');
                    if (loadMore) loadMore.click();
                })();
                """
            ],

            # Wait for new content to appear
            wait_for="js:() => document.querySelectorAll('.item').length > 20",

            # Additional timing controls
            page_timeout=60000,  # 60 second timeout
            delay_before_return_html=2.0,  # Wait before final capture

            # Handle overlays automatically
            remove_overlay_elements=True,
            magic=True,  # Auto-handle common popup patterns

            # Simulate human behavior
            simulate_user=True,
            override_navigator=True
        )

        result = await crawler.arun("https://example.com/dynamic", config=config)

        if result.success:
            print(f"Interactive crawl completed: {len(result.cleaned_html)} chars")

# Form interaction example
async def form_interaction_example():
    config = CrawlerRunConfig(
        js_code="""
            // Fill search form
            document.querySelector('#search-input').value = 'machine learning';
            document.querySelector('#category-select').value = 'technology';
            document.querySelector('#search-form').submit();
        """,
        wait_for="css:.search-results",
        session_id="search_session"
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://example.com/search", config=config)
        print("Search completed, results loaded")
```

### Local File & Raw HTML Processing

```python
# Handle different input types: URLs, local files, raw HTML
async def handle_different_inputs():
    async with AsyncWebCrawler() as crawler:
        # 1. Regular web URL
        result1 = await crawler.arun("https://example.com")

        # 2. Local HTML file
        local_file_path = "/path/to/file.html"
        result2 = await crawler.arun(f"file://{local_file_path}")

        # 3. Raw HTML content
        raw_html = "<html><body><h1>Test Content</h1><p>Sample text</p></body></html>"
        result3 = await crawler.arun(f"raw:{raw_html}")

        # All return the same CrawlResult structure
        for i, result in enumerate([result1, result2, result3], 1):
            if result.success:
                print(f"Input {i}: {len(result.markdown)} chars of markdown")

# Save and re-process HTML example
async def save_and_reprocess():
    async with AsyncWebCrawler() as crawler:
        # Original crawl
        result = await crawler.arun("https://example.com")

        if result.success:
            # Save HTML to file
            with open("saved_page.html", "w", encoding="utf-8") as f:
                f.write(result.html)

            # Re-process from file
            file_result = await crawler.arun("file://./saved_page.html")

            # Process as raw HTML
            raw_result = await crawler.arun(f"raw:{result.html}")

            # Verify consistency
            assert len(result.markdown) == len(file_result.markdown) == len(raw_result.markdown)
            print("✅ All processing methods produced identical results")
```

### Advanced Link & Media Handling

```python
# Comprehensive link and media extraction with filtering
async def advanced_link_media_handling():
    config = CrawlerRunConfig(
        # Link filtering
        exclude_external_links=False,  # Keep external links for analysis
        exclude_social_media_links=True,
        exclude_domains=["ads.com", "tracker.io", "spammy.net"],

        # Media handling
        exclude_external_images=True,
        image_score_threshold=5,  # Only high-quality images
        table_score_threshold=7,  # Only well-structured tables
        wait_for_images=True,

        # Capture additional formats
        screenshot=True,
        pdf=True,
        capture_mhtml=True  # Full page archive
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://example.com", config=config)

        if result.success:
            # Analyze links
            internal_links = result.links.get("internal", [])
            external_links = result.links.get("external", [])
            print(f"Links: {len(internal_links)} internal, {len(external_links)} external")

            # Analyze media
            images = result.media.get("images", [])
            tables = result.media.get("tables", [])
            print(f"Media: {len(images)} images, {len(tables)} tables")

            # High-quality images only
            quality_images = [img for img in images if img.get("score", 0) >= 5]
            print(f"High-quality images: {len(quality_images)}")

            # Table analysis
            for i, table in enumerate(tables[:2]):
                print(f"Table {i+1}: {len(table.get('headers', []))} columns, {len(table.get('rows', []))} rows")

            # Save captured files
            if result.screenshot:
                import base64
                with open("page_screenshot.png", "wb") as f:
                    f.write(base64.b64decode(result.screenshot))

            if result.pdf:
                with open("page.pdf", "wb") as f:
                    f.write(result.pdf)

            if result.mhtml:
                with open("page_archive.mhtml", "w", encoding="utf-8") as f:
                    f.write(result.mhtml)

            print("Additional formats saved: screenshot, PDF, MHTML archive")
```

### Performance & Resource Management

```python
# Optimize performance for large-scale crawling
async def performance_optimized_crawling():
    # Lightweight browser config
    browser_config = BrowserConfig(
        headless=True,
        text_mode=True,  # Disable images for speed
        light_mode=True,  # Reduce background features
        extra_args=["--disable-extensions", "--no-sandbox"]
    )

    # Efficient crawl config
    config = CrawlerRunConfig(
        # Content filtering for speed
        excluded_tags=["script", "style", "nav", "footer"],
        exclude_external_links=True,
        exclude_all_images=True,  # Remove all images for max speed
        word_count_threshold=50,

        # Timing optimizations
        page_timeout=30000,  # Faster timeout
        delay_before_return_html=0.1,

        # Resource monitoring
        capture_network_requests=False,  # Disable unless needed
        capture_console_messages=False,

        # Cache for repeated URLs
        cache_mode=CacheMode.ENABLED
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        urls = ["https://example.com/page1", "https://example.com/page2", "https://example.com/page3"]

        # Efficient batch processing
        batch_config = config.clone(
            stream=True,  # Stream results as they complete
            semaphore_count=3  # Control concurrency
        )

        async for result in await crawler.arun_many(urls, config=batch_config):
            if result.success:
                print(f"✅ {result.url}: {len(result.markdown)} chars")
            else:
                print(f"❌ {result.url}: {result.error_message}")
```


**📖 Learn more:** [Complete Parameter Reference](https://docs.crawl4ai.com/api/parameters/), [Content Filtering](https://docs.crawl4ai.com/core/markdown-generation/), [Session Management](https://docs.crawl4ai.com/advanced/session-management/), [Network Capture](https://docs.crawl4ai.com/advanced/network-console-capture/)

**📖 Learn more:** [Hooks & Authentication](https://docs.crawl4ai.com/advanced/hooks-auth/), [Session Management](https://docs.crawl4ai.com/advanced/session-management/), [Network Monitoring](https://docs.crawl4ai.com/advanced/network-console-capture/), [Page Interaction](https://docs.crawl4ai.com/core/page-interaction/), [File Downloads](https://docs.crawl4ai.com/advanced/file-downloading/)

---


## Configuration Objects - Diagrams & Workflows
Component ID: config_objects
Context Type: reasoning
Estimated tokens: 9,795

## Configuration Objects and System Architecture

Visual representations of Crawl4AI's configuration system, object relationships, and data flow patterns.

### Configuration Object Relationships

```mermaid
classDiagram
    class BrowserConfig {
        +browser_type: str
        +headless: bool
        +viewport_width: int
        +viewport_height: int
        +proxy: str
        +user_agent: str
        +cookies: list
        +headers: dict
        +clone() BrowserConfig
        +to_dict() dict
    }

    class CrawlerRunConfig {
        +cache_mode: CacheMode
        +extraction_strategy: ExtractionStrategy
        +markdown_generator: MarkdownGenerator
        +js_code: list
        +wait_for: str
        +screenshot: bool
        +session_id: str
        +clone() CrawlerRunConfig
        +dump() dict
    }

    class LLMConfig {
        +provider: str
        +api_token: str
        +base_url: str
        +temperature: float
        +max_tokens: int
        +clone() LLMConfig
        +to_dict() dict
    }

    class CrawlResult {
        +url: str
        +success: bool
        +html: str
        +cleaned_html: str
        +markdown: MarkdownGenerationResult
        +extracted_content: str
        +media: dict
        +links: dict
        +screenshot: str
        +pdf: bytes
    }

    class AsyncWebCrawler {
        +config: BrowserConfig
        +arun() CrawlResult
    }

    AsyncWebCrawler --> BrowserConfig : uses
    AsyncWebCrawler --> CrawlerRunConfig : accepts
    CrawlerRunConfig --> LLMConfig : contains
    AsyncWebCrawler --> CrawlResult : returns

    note for BrowserConfig "Controls browser\nenvironment and behavior"
    note for CrawlerRunConfig "Controls individual\ncrawl operations"
    note for LLMConfig "Configures LLM\nproviders and parameters"
    note for CrawlResult "Contains all crawl\noutputs and metadata"
```

### Configuration Decision Flow

```mermaid
flowchart TD
    A[Start Configuration] --> B{Use Case Type?}

    B -->|Simple Web Scraping| C[Basic Config Pattern]
    B -->|Data Extraction| D[Extraction Config Pattern]
    B -->|Stealth Crawling| E[Stealth Config Pattern]
    B -->|High Performance| F[Performance Config Pattern]

    C --> C1[BrowserConfig: headless=True]
    C --> C2[CrawlerRunConfig: basic options]
    C1 --> C3[No LLMConfig needed]
    C2 --> C3
    C3 --> G[Simple Crawling Ready]

    D --> D1[BrowserConfig: standard setup]
    D --> D2[CrawlerRunConfig: with extraction_strategy]
    D --> D3[LLMConfig: for LLM extraction]
    D1 --> D4[Advanced Extraction Ready]
    D2 --> D4
    D3 --> D4

    E --> E1[BrowserConfig: proxy + user_agent]
    E --> E2[CrawlerRunConfig: simulate_user=True]
    E1 --> E3[Stealth Crawling Ready]
    E2 --> E3

    F --> F1[BrowserConfig: lightweight]
    F --> F2[CrawlerRunConfig: caching + concurrent]
    F1 --> F3[High Performance Ready]
    F2 --> F3

    G --> H[Execute Crawl]
    D4 --> H
    E3 --> H
    F3 --> H

    H --> I[Get CrawlResult]

    style A fill:#e1f5fe
    style I fill:#c8e6c9
    style G fill:#fff3e0
    style D4 fill:#f3e5f5
    style E3 fill:#ffebee
    style F3 fill:#e8f5e8
```

### Configuration Lifecycle Sequence

```mermaid
sequenceDiagram
    participant User
    participant BrowserConfig as Browser Config
    participant CrawlerConfig as Crawler Config
    participant LLMConfig as LLM Config
    participant Crawler as AsyncWebCrawler
    participant Browser as Browser Instance
    participant Result as CrawlResult

    User->>BrowserConfig: Create with browser settings
    User->>CrawlerConfig: Create with crawl options
    User->>LLMConfig: Create with LLM provider

    User->>Crawler: Initialize with BrowserConfig
    Crawler->>Browser: Launch browser with config
    Browser-->>Crawler: Browser ready

    User->>Crawler: arun(url, CrawlerConfig)
    Crawler->>Crawler: Apply CrawlerConfig settings

    alt LLM Extraction Needed
        Crawler->>LLMConfig: Get LLM settings
        LLMConfig-->>Crawler: Provider configuration
    end

    Crawler->>Browser: Navigate with settings
    Browser->>Browser: Apply page interactions
    Browser->>Browser: Execute JavaScript if specified
    Browser->>Browser: Wait for conditions

    Browser-->>Crawler: Page content ready
    Crawler->>Crawler: Process content per config
    Crawler->>Result: Create CrawlResult

    Result-->>User: Return complete result

    Note over User,Result: Configuration objects control every aspect
```

### BrowserConfig Parameter Flow

```mermaid
graph TB
    subgraph "BrowserConfig Parameters"
        A[browser_type] --> A1[chromium/firefox/webkit]
        B[headless] --> B1[true: invisible / false: visible]
        C[viewport] --> C1[width x height dimensions]
        D[proxy] --> D1[proxy server configuration]
        E[user_agent] --> E1[browser identification string]
        F[cookies] --> F1[session authentication]
        G[headers] --> G1[HTTP request headers]
        H[extra_args] --> H1[browser command line flags]
    end

    subgraph "Browser Instance"
        I[Playwright Browser]
        J[Browser Context]
        K[Page Instance]
    end

    A1 --> I
    B1 --> I
    C1 --> J
    D1 --> J
    E1 --> J
    F1 --> J
    G1 --> J
    H1 --> I

    I --> J
    J --> K

    style I fill:#e3f2fd
    style J fill:#f3e5f5
    style K fill:#e8f5e8
```

### CrawlerRunConfig Category Breakdown

```mermaid
mindmap
  root((CrawlerRunConfig))
    Content Processing
      word_count_threshold
      css_selector
      target_elements
      excluded_tags
      markdown_generator
      extraction_strategy
    Page Navigation
      wait_until
      page_timeout
      wait_for
      wait_for_images
      delay_before_return_html
    Page Interaction
      js_code
      scan_full_page
      simulate_user
      magic
      remove_overlay_elements
    Caching Session
      cache_mode
      session_id
      shared_data
    Media Output
      screenshot
      pdf
      capture_mhtml
      image_score_threshold
    Link Filtering
      exclude_external_links
      exclude_domains
      exclude_social_media_links
```

### LLM Provider Selection Flow

```mermaid
flowchart TD
    A[Need LLM Processing?] --> B{Provider Type?}

    B -->|Cloud API| C{Which Service?}
    B -->|Local Model| D[Local Setup]
    B -->|Custom Endpoint| E[Custom Config]

    C -->|OpenAI| C1[OpenAI GPT Models]
    C -->|Anthropic| C2[Claude Models]
    C -->|Google| C3[Gemini Models]
    C -->|Groq| C4[Fast Inference]

    D --> D1[Ollama Setup]
    E --> E1[Custom base_url]

    C1 --> F1[LLMConfig with OpenAI settings]
    C2 --> F2[LLMConfig with Anthropic settings]
    C3 --> F3[LLMConfig with Google settings]
    C4 --> F4[LLMConfig with Groq settings]
    D1 --> F5[LLMConfig with Ollama settings]
    E1 --> F6[LLMConfig with custom settings]

    F1 --> G[Use in Extraction Strategy]
    F2 --> G
    F3 --> G
    F4 --> G
    F5 --> G
    F6 --> G

    style A fill:#e1f5fe
    style G fill:#c8e6c9
```

### CrawlResult Structure and Data Flow

```mermaid
graph TB
    subgraph "CrawlResult Output"
        A[Basic Info]
        B[HTML Content]
        C[Markdown Output]
        D[Extracted Data]
        E[Media Files]
        F[Metadata]
    end

    subgraph "Basic Info Details"
        A --> A1[url: final URL]
        A --> A2[success: boolean]
        A --> A3[status_code: HTTP status]
        A --> A4[error_message: if failed]
    end

    subgraph "HTML Content Types"
        B --> B1[html: raw HTML]
        B --> B2[cleaned_html: processed]
        B --> B3[fit_html: filtered content]
    end

    subgraph "Markdown Variants"
        C --> C1[raw_markdown: basic conversion]
        C --> C2[markdown_with_citations: with refs]
        C --> C3[fit_markdown: filtered content]
        C --> C4[references_markdown: citation list]
    end

    subgraph "Extracted Content"
        D --> D1[extracted_content: JSON string]
        D --> D2[From CSS extraction]
        D --> D3[From LLM extraction]
        D --> D4[From XPath extraction]
    end

    subgraph "Media and Links"
        E --> E1[images: list with scores]
        E --> E2[videos: media content]
        E --> E3[internal_links: same domain]
        E --> E4[external_links: other domains]
    end

    subgraph "Generated Files"
        F --> F1[screenshot: base64 PNG]
        F --> F2[pdf: binary PDF data]
        F --> F3[mhtml: archive format]
        F --> F4[ssl_certificate: cert info]
    end

    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#ffebee
    style F fill:#f1f8e9
```

### Configuration Pattern State Machine

```mermaid
stateDiagram-v2
    [*] --> ConfigCreation

    ConfigCreation --> BasicConfig: Simple use case
    ConfigCreation --> AdvancedConfig: Complex requirements
    ConfigCreation --> TemplateConfig: Use predefined pattern

    BasicConfig --> Validation: Check parameters
    AdvancedConfig --> Validation: Check parameters
    TemplateConfig --> Validation: Check parameters

    Validation --> Invalid: Missing required fields
    Validation --> Valid: All parameters correct

    Invalid --> ConfigCreation: Fix and retry

    Valid --> InUse: Passed to crawler
    InUse --> Cloning: Need variation
    InUse --> Serialization: Save configuration
    InUse --> Complete: Crawl finished

    Cloning --> Modified: clone() with updates
    Modified --> Valid: Validate changes

    Serialization --> Stored: dump() to dict
    Stored --> Restoration: load() from dict
    Restoration --> Valid: Recreate config object

    Complete --> [*]

    note right of BasicConfig : Minimal required settings
    note right of AdvancedConfig : Full feature configuration
    note right of TemplateConfig : Pre-built patterns
```

### Configuration Integration Architecture

```mermaid
graph TB
    subgraph "User Layer"
        U1[Configuration Creation]
        U2[Parameter Selection]
        U3[Pattern Application]
    end

    subgraph "Configuration Layer"
        C1[BrowserConfig]
        C2[CrawlerRunConfig]
        C3[LLMConfig]
        C4[Config Validation]
        C5[Config Cloning]
    end

    subgraph "Crawler Engine"
        E1[Browser Management]
        E2[Page Navigation]
        E3[Content Processing]
        E4[Extraction Pipeline]
        E5[Result Generation]
    end

    subgraph "Output Layer"
        O1[CrawlResult Assembly]
        O2[Data Formatting]
        O3[File Generation]
        O4[Metadata Collection]
    end

    U1 --> C1
    U2 --> C2
    U3 --> C3

    C1 --> C4
    C2 --> C4
    C3 --> C4

    C4 --> E1
    C2 --> E2
    C2 --> E3
    C3 --> E4

    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> E5

    E5 --> O1
    O1 --> O2
    O2 --> O3
    O3 --> O4

    C5 -.-> C1
    C5 -.-> C2
    C5 -.-> C3

    style U1 fill:#e1f5fe
    style C4 fill:#fff3e0
    style E4 fill:#f3e5f5
    style O4 fill:#c8e6c9
```

### Configuration Best Practices Flow

```mermaid
flowchart TD
    A[Configuration Planning] --> B{Performance Priority?}

    B -->|Speed| C[Fast Config Pattern]
    B -->|Quality| D[Comprehensive Config Pattern]
    B -->|Stealth| E[Stealth Config Pattern]
    B -->|Balanced| F[Standard Config Pattern]

    C --> C1[Enable caching]
    C --> C2[Disable heavy features]
    C --> C3[Use text_mode]
    C1 --> G[Apply Configuration]
    C2 --> G
    C3 --> G

    D --> D1[Enable all processing]
    D --> D2[Use content filters]
    D --> D3[Capture everything]
    D1 --> G
    D2 --> G
    D3 --> G

    E --> E1[Rotate user agents]
    E --> E2[Use proxies]
    E --> E3[Simulate human behavior]
    E1 --> G
    E2 --> G
    E3 --> G

    F --> F1[Balanced timeouts]
    F --> F2[Selective processing]
    F --> F3[Smart caching]
    F1 --> G
    F2 --> G
    F3 --> G

    G --> H[Test Configuration]
    H --> I{Results Satisfactory?}

    I -->|Yes| J[Production Ready]
    I -->|No| K[Adjust Parameters]

    K --> L[Clone and Modify]
    L --> H

    J --> M[Deploy with Confidence]

    style A fill:#e1f5fe
    style J fill:#c8e6c9
    style M fill:#e8f5e8
```

## Advanced Configuration Workflows and Patterns

Visual representations of advanced Crawl4AI configuration strategies, proxy management, session handling, and identity-based crawling patterns.

### User Agent and Anti-Detection Strategy Flow

```mermaid
flowchart TD
    A[Start Configuration] --> B{Detection Avoidance Needed?}

    B -->|No| C[Standard User Agent]
    B -->|Yes| D[Anti-Detection Strategy]

    C --> C1[Static user_agent string]
    C1 --> Z[Basic Configuration]

    D --> E{User Agent Strategy}
    E -->|Random| F[user_agent_mode: random]
    E -->|Static Custom| G[Custom user_agent string]
    E -->|Platform Specific| H[Generator Config]

    F --> I[Configure Generator]
    H --> I
    I --> I1[Platform: windows/macos/linux]
    I1 --> I2[Browser: chrome/firefox/safari]
    I2 --> I3[Device: desktop/mobile/tablet]

    G --> J[Behavioral Simulation]
    I3 --> J

    J --> K{Enable Simulation?}
    K -->|Yes| L[simulate_user: True]
    K -->|No| M[Standard Behavior]

    L --> N[override_navigator: True]
    N --> O[Configure Delays]
    O --> O1[mean_delay: 1.5]
    O1 --> O2[max_range: 2.0]
    O2 --> P[Magic Mode]

    M --> P
    P --> Q{Auto-Handle Patterns?}
    Q -->|Yes| R[magic: True]
    Q -->|No| S[Manual Handling]

    R --> T[Complete Anti-Detection Setup]
    S --> T
    Z --> T

    style D fill:#ffeb3b
    style T fill:#c8e6c9
    style L fill:#ff9800
    style R fill:#9c27b0
```

### Proxy Configuration and Rotation Architecture

```mermaid
graph TB
    subgraph "Proxy Configuration Types"
        A[Single Proxy] --> A1[ProxyConfig object]
        B[Proxy String] --> B1[from_string method]
        C[Environment Proxies] --> C1[from_env method]
        D[Multiple Proxies] --> D1[ProxyRotationStrategy]
    end

    subgraph "ProxyConfig Structure"
        A1 --> E[server: URL]
        A1 --> F[username: auth]
        A1 --> G[password: auth]
        A1 --> H[ip: extracted]
    end

    subgraph "Rotation Strategies"
        D1 --> I[round_robin]
        D1 --> J[random]
        D1 --> K[least_used]
        D1 --> L[failure_aware]
    end

    subgraph "Configuration Flow"
        M[CrawlerRunConfig] --> N[proxy_config]
        M --> O[proxy_rotation_strategy]
        N --> P[Single Proxy Usage]
        O --> Q[Multi-Proxy Rotation]
    end

    subgraph "Runtime Behavior"
        P --> R[All requests use same proxy]
        Q --> S[Requests rotate through proxies]
        S --> T[Health monitoring]
        T --> U[Automatic failover]
    end

    style A1 fill:#e3f2fd
    style D1 fill:#f3e5f5
    style M fill:#e8f5e8
    style T fill:#fff3e0
```

### Content Selection Strategy Comparison

```mermaid
sequenceDiagram
    participant Browser
    participant HTML as Raw HTML
    participant CSS as css_selector
    participant Target as target_elements
    participant Processor as Content Processor
    participant Output

    Note over Browser,Output: css_selector Strategy
    Browser->>HTML: Load complete page
    HTML->>CSS: Apply css_selector
    CSS->>CSS: Extract matching elements only
    CSS->>Processor: Process subset HTML
    Processor->>Output: Markdown + Extraction from subset

    Note over Browser,Output: target_elements Strategy
    Browser->>HTML: Load complete page
    HTML->>Processor: Process entire page
    Processor->>Target: Focus on target_elements
    Target->>Target: Extract from specified elements
    Processor->>Output: Full page links/media + targeted content

    Note over CSS,Target: Key Difference
    Note over CSS: Affects entire processing pipeline
    Note over Target: Affects only content extraction
```

### Advanced wait_for Conditions Decision Tree

```mermaid
flowchart TD
    A[Configure wait_for] --> B{Condition Type?}

    B -->|CSS Element| C[CSS Selector Wait]
    B -->|JavaScript Condition| D[JS Expression Wait]
    B -->|Complex Logic| E[Custom JS Function]
    B -->|No Wait| F[Default domcontentloaded]

    C --> C1["wait_for: 'css:.element'"]
    C1 --> C2[Element appears in DOM]
    C2 --> G[Continue Processing]

    D --> D1["wait_for: 'js:() => condition'"]
    D1 --> D2[JavaScript returns true]
    D2 --> G

    E --> E1[Complex JS Function]
    E1 --> E2{Multiple Conditions}
    E2 -->|AND Logic| E3[All conditions true]
    E2 -->|OR Logic| E4[Any condition true]
    E2 -->|Custom Logic| E5[User-defined logic]

    E3 --> G
    E4 --> G
    E5 --> G

    F --> G

    G --> H{Timeout Reached?}
    H -->|No| I[Page Ready]
    H -->|Yes| J[Timeout Error]

    I --> K[Begin Content Extraction]
    J --> L[Handle Error/Retry]

    style C1 fill:#e8f5e8
    style D1 fill:#fff3e0
    style E1 fill:#ffeb3b
    style I fill:#c8e6c9
    style J fill:#ffcdd2
```

### Session Management Lifecycle

```mermaid
stateDiagram-v2
    [*] --> SessionCreate

    SessionCreate --> SessionActive: session_id provided
    SessionCreate --> OneTime: no session_id

    SessionActive --> BrowserLaunch: First arun() call
    BrowserLaunch --> PageLoad: Navigate to URL
    PageLoad --> JSExecution: Execute js_code
    JSExecution --> ContentExtract: Extract content
    ContentExtract --> SessionHold: Keep session alive

    SessionHold --> ReuseSession: Subsequent arun() calls
    ReuseSession --> JSOnlyMode: js_only=True
    ReuseSession --> NewNavigation: js_only=False

    JSOnlyMode --> JSExecution: Execute JS in existing page
    NewNavigation --> PageLoad: Navigate to new URL

    SessionHold --> SessionKill: kill_session() called
    SessionHold --> SessionTimeout: Timeout reached
    SessionHold --> SessionError: Error occurred

    SessionKill --> SessionCleanup
    SessionTimeout --> SessionCleanup
    SessionError --> SessionCleanup
    SessionCleanup --> [*]

    OneTime --> BrowserLaunch
    ContentExtract --> OneTimeCleanup: No session_id
    OneTimeCleanup --> [*]

    note right of SessionActive : Persistent browser context
    note right of JSOnlyMode : Reuse existing page
    note right of OneTime : Temporary browser instance
```

### Identity-Based Crawling Configuration Matrix

```mermaid
graph TD
    subgraph "Geographic Identity"
        A[Geolocation] --> A1[latitude/longitude]
        A2[Timezone] --> A3[timezone_id]
        A4[Locale] --> A5[language/region]
    end

    subgraph "Browser Identity"
        B[User Agent] --> B1[Platform fingerprint]
        B2[Navigator Properties] --> B3[override_navigator]
        B4[Headers] --> B5[Accept-Language]
    end

    subgraph "Behavioral Identity"
        C[Mouse Simulation] --> C1[simulate_user]
        C2[Timing Patterns] --> C3[mean_delay/max_range]
        C4[Interaction Patterns] --> C5[Human-like behavior]
    end

    subgraph "Configuration Integration"
        D[CrawlerRunConfig] --> A
        D --> B
        D --> C

        D --> E[Complete Identity Profile]

        E --> F[Geographic Consistency]
        E --> G[Browser Consistency]
        E --> H[Behavioral Consistency]
    end

    F --> I[Paris, France Example]
    I --> I1[locale: fr-FR]
    I --> I2[timezone: Europe/Paris]
    I --> I3[geolocation: 48.8566, 2.3522]

    G --> J[Windows Chrome Example]
    J --> J1[platform: windows]
    J --> J2[browser: chrome]
    J --> J3[user_agent: matching pattern]

    H --> K[Human Simulation]
    K --> K1[Random delays]
    K --> K2[Mouse movements]
    K --> K3[Navigation patterns]

    style E fill:#ff9800
    style I fill:#e3f2fd
    style J fill:#f3e5f5
    style K fill:#e8f5e8
```

### Multi-Step Crawling Sequence Flow

```mermaid
sequenceDiagram
    participant User
    participant Crawler
    participant Session as Browser Session
    participant Page1 as Login Page
    participant Page2 as Dashboard
    participant Page3 as Data Pages

    User->>Crawler: Step 1 - Login
    Crawler->>Session: Create session_id="user_session"
    Session->>Page1: Navigate to login
    Page1->>Page1: Execute login JS
    Page1->>Page1: Wait for dashboard redirect
    Page1-->>Crawler: Login complete

    User->>Crawler: Step 2 - Navigate dashboard
    Note over Crawler,Session: Reuse existing session
    Crawler->>Session: js_only=True (no page reload)
    Session->>Page2: Execute navigation JS
    Page2->>Page2: Wait for data table
    Page2-->>Crawler: Dashboard ready

    User->>Crawler: Step 3 - Extract data pages
    loop For each page 1-5
        Crawler->>Session: js_only=True
        Session->>Page3: Click page button
        Page3->>Page3: Wait for page active
        Page3->>Page3: Extract content
        Page3-->>Crawler: Page data
    end

    User->>Crawler: Cleanup
    Crawler->>Session: kill_session()
    Session-->>Crawler: Session destroyed
```

### Configuration Import and Usage Patterns

```mermaid
graph LR
    subgraph "Main Package Imports"
        A[crawl4ai] --> A1[AsyncWebCrawler]
        A --> A2[BrowserConfig]
        A --> A3[CrawlerRunConfig]
        A --> A4[LLMConfig]
        A --> A5[CacheMode]
        A --> A6[ProxyConfig]
        A --> A7[GeolocationConfig]
    end

    subgraph "Strategy Imports"
        A --> B1[JsonCssExtractionStrategy]
        A --> B2[LLMExtractionStrategy]
        A --> B3[DefaultMarkdownGenerator]
        A --> B4[PruningContentFilter]
        A --> B5[RegexChunking]
    end

    subgraph "Configuration Assembly"
        C[Configuration Builder] --> A2
        C --> A3
        C --> A4

        A2 --> D[Browser Environment]
        A3 --> E[Crawl Behavior]
        A4 --> F[LLM Integration]

        E --> B1
        E --> B2
        E --> B3
        E --> B4
        E --> B5
    end

    subgraph "Runtime Flow"
        G[Crawler Instance] --> D
        G --> H[Execute Crawl]
        H --> E
        H --> F
        H --> I[CrawlResult]
    end

    style A fill:#e3f2fd
    style C fill:#fff3e0
    style G fill:#e8f5e8
    style I fill:#c8e6c9
```

### Advanced Configuration Decision Matrix

```mermaid
flowchart TD
    A[Advanced Configuration Needed] --> B{Primary Use Case?}

    B -->|Bot Detection Avoidance| C[Anti-Detection Setup]
    B -->|Geographic Simulation| D[Identity-Based Config]
    B -->|Multi-Step Workflows| E[Session Management]
    B -->|Network Reliability| F[Proxy Configuration]
    B -->|Content Precision| G[Selector Strategy]

    C --> C1[Random User Agents]
    C --> C2[Behavioral Simulation]
    C --> C3[Navigator Override]
    C --> C4[Magic Mode]

    D --> D1[Geolocation Setup]
    D --> D2[Locale Configuration]
    D --> D3[Timezone Setting]
    D --> D4[Browser Fingerprinting]

    E --> E1[Session ID Management]
    E --> E2[JS-Only Navigation]
    E --> E3[Shared Data Context]
    E --> E4[Session Cleanup]

    F --> F1[Single Proxy]
    F --> F2[Proxy Rotation]
    F --> F3[Failover Strategy]
    F --> F4[Health Monitoring]

    G --> G1[css_selector for Subset]
    G --> G2[target_elements for Focus]
    G --> G3[excluded_selector for Removal]
    G --> G4[Hierarchical Selection]

    C1 --> H[Production Configuration]
    C2 --> H
    C3 --> H
    C4 --> H
    D1 --> H
    D2 --> H
    D3 --> H
    D4 --> H
    E1 --> H
    E2 --> H
    E3 --> H
    E4 --> H
    F1 --> H
    F2 --> H
    F3 --> H
    F4 --> H
    G1 --> H
    G2 --> H
    G3 --> H
    G4 --> H

    style H fill:#c8e6c9
    style C fill:#ff9800
    style D fill:#9c27b0
    style E fill:#2196f3
    style F fill:#4caf50
    style G fill:#ff5722
```

## Advanced Features Workflows and Architecture

Visual representations of advanced crawling capabilities, session management, hooks system, and performance optimization strategies.

### File Download Workflow

```mermaid
sequenceDiagram
    participant User
    participant Crawler
    participant Browser
    participant FileSystem
    participant Page

    User->>Crawler: Configure downloads_path
    Crawler->>Browser: Create context with download handling
    Browser-->>Crawler: Context ready

    Crawler->>Page: Navigate to target URL
    Page-->>Crawler: Page loaded

    Crawler->>Page: Execute download JavaScript
    Page->>Page: Find download links (.pdf, .zip, etc.)

    loop For each download link
        Page->>Browser: Click download link
        Browser->>FileSystem: Save file to downloads_path
        FileSystem-->>Browser: File saved
        Browser-->>Page: Download complete
    end

    Page-->>Crawler: All downloads triggered
    Crawler->>FileSystem: Check downloaded files
    FileSystem-->>Crawler: List of file paths
    Crawler-->>User: CrawlResult with downloaded_files[]

    Note over User,FileSystem: Files available in downloads_path
```

### Hooks Execution Flow

```mermaid
flowchart TD
    A[Start Crawl] --> B[on_browser_created Hook]
    B --> C[Browser Instance Created]
    C --> D[on_page_context_created Hook]
    D --> E[Page & Context Setup]
    E --> F[before_goto Hook]
    F --> G[Navigate to URL]
    G --> H[after_goto Hook]
    H --> I[Page Loaded]
    I --> J[before_retrieve_html Hook]
    J --> K[Extract HTML Content]
    K --> L[Return CrawlResult]

    subgraph "Hook Capabilities"
        B1[Route Filtering]
        B2[Authentication]
        B3[Custom Headers]
        B4[Viewport Setup]
        B5[Content Manipulation]
    end

    D --> B1
    F --> B2
    F --> B3
    D --> B4
    J --> B5

    style A fill:#e1f5fe
    style L fill:#c8e6c9
    style B fill:#fff3e0
    style D fill:#f3e5f5
    style F fill:#e8f5e8
    style H fill:#fce4ec
    style J fill:#fff9c4
```

### Session Management State Machine

```mermaid
stateDiagram-v2
    [*] --> SessionCreated: session_id provided

    SessionCreated --> PageLoaded: Initial arun()
    PageLoaded --> JavaScriptExecution: js_code executed
    JavaScriptExecution --> ContentUpdated: DOM modified
    ContentUpdated --> NextOperation: js_only=True

    NextOperation --> JavaScriptExecution: More interactions
    NextOperation --> SessionMaintained: Keep session alive
    NextOperation --> SessionClosed: kill_session()

    SessionMaintained --> PageLoaded: Navigate to new URL
    SessionMaintained --> JavaScriptExecution: Continue interactions

    SessionClosed --> [*]: Session terminated

    note right of SessionCreated
        Browser tab created
        Context preserved
    end note

    note right of ContentUpdated
        State maintained
        Cookies preserved
        Local storage intact
    end note

    note right of SessionClosed
        Clean up resources
        Release browser tab
    end note
```

### Lazy Loading & Dynamic Content Strategy

```mermaid
flowchart TD
    A[Page Load] --> B{Content Type?}

    B -->|Static Content| C[Standard Extraction]
    B -->|Lazy Loaded| D[Enable scan_full_page]
    B -->|Infinite Scroll| E[Custom Scroll Strategy]
    B -->|Load More Button| F[JavaScript Interaction]

    D --> D1[Automatic Scrolling]
    D1 --> D2[Wait for Images]
    D2 --> D3[Content Stabilization]

    E --> E1[Detect Scroll Triggers]
    E1 --> E2[Progressive Loading]
    E2 --> E3[Monitor Content Changes]

    F --> F1[Find Load More Button]
    F1 --> F2[Click and Wait]
    F2 --> F3{More Content?}
    F3 -->|Yes| F1
    F3 -->|No| G[Complete Extraction]

    D3 --> G
    E3 --> G
    C --> G

    G --> H[Return Enhanced Content]

    subgraph "Optimization Techniques"
        I[exclude_external_images]
        J[image_score_threshold]
        K[wait_for selectors]
        L[scroll_delay tuning]
    end

    D --> I
    E --> J
    F --> K
    D1 --> L

    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style D fill:#fff3e0
    style E fill:#f3e5f5
    style F fill:#e8f5e8
```

### Network & Console Monitoring Architecture

```mermaid
graph TB
    subgraph "Browser Context"
        A[Web Page] --> B[Network Requests]
        A --> C[Console Messages]
        A --> D[Resource Loading]
    end

    subgraph "Monitoring Layer"
        B --> E[Request Interceptor]
        C --> F[Console Listener]
        D --> G[Resource Monitor]

        E --> H[Request Events]
        E --> I[Response Events]
        E --> J[Failure Events]

        F --> K[Log Messages]
        F --> L[Error Messages]
        F --> M[Warning Messages]
    end

    subgraph "Data Collection"
        H --> N[Request Details]
        I --> O[Response Analysis]
        J --> P[Failure Tracking]

        K --> Q[Debug Information]
        L --> R[Error Analysis]
        M --> S[Performance Insights]
    end

    subgraph "Output Aggregation"
        N --> T[network_requests Array]
        O --> T
        P --> T

        Q --> U[console_messages Array]
        R --> U
        S --> U
    end

    T --> V[CrawlResult]
    U --> V

    style V fill:#c8e6c9
    style E fill:#fff3e0
    style F fill:#f3e5f5
    style T fill:#e8f5e8
    style U fill:#fce4ec
```

### Multi-Step Workflow Sequence

```mermaid
sequenceDiagram
    participant User
    participant Crawler
    participant Session
    participant Page
    participant Server

    User->>Crawler: Step 1 - Initial load
    Crawler->>Session: Create session_id
    Session->>Page: New browser tab
    Page->>Server: GET /step1
    Server-->>Page: Page content
    Page-->>Crawler: Content ready
    Crawler-->>User: Result 1

    User->>Crawler: Step 2 - Navigate (js_only=true)
    Crawler->>Session: Reuse existing session
    Session->>Page: Execute JavaScript
    Page->>Page: Click next button
    Page->>Server: Navigate to /step2
    Server-->>Page: New content
    Page-->>Crawler: Updated content
    Crawler-->>User: Result 2

    User->>Crawler: Step 3 - Form submission
    Crawler->>Session: Continue session
    Session->>Page: Execute form JS
    Page->>Page: Fill form fields
    Page->>Server: POST form data
    Server-->>Page: Results page
    Page-->>Crawler: Final content
    Crawler-->>User: Result 3

    User->>Crawler: Cleanup
    Crawler->>Session: kill_session()
    Session->>Page: Close tab
    Session-->>Crawler: Session terminated

    Note over User,Server: State preserved across steps
    Note over Session: Cookies, localStorage maintained
```

### SSL Certificate Analysis Flow

```mermaid
flowchart LR
    A[Enable SSL Fetch] --> B[HTTPS Connection]
    B --> C[Certificate Retrieval]
    C --> D[Certificate Analysis]

    D --> E[Basic Info]
    D --> F[Validity Check]
    D --> G[Chain Verification]
    D --> H[Security Assessment]

    E --> E1[Issuer Details]
    E --> E2[Subject Information]
    E --> E3[Serial Number]

    F --> F1[Not Before Date]
    F --> F2[Not After Date]
    F --> F3[Expiration Warning]

    G --> G1[Root CA]
    G --> G2[Intermediate Certs]
    G --> G3[Trust Path]

    H --> H1[Key Length]
    H --> H2[Signature Algorithm]
    H --> H3[Vulnerabilities]

    subgraph "Export Formats"
        I[JSON Format]
        J[PEM Format]
        K[DER Format]
    end

    E1 --> I
    F1 --> I
    G1 --> I
    H1 --> I

    I --> J
    J --> K

    style A fill:#e1f5fe
    style D fill:#fff3e0
    style I fill:#e8f5e8
    style J fill:#f3e5f5
    style K fill:#fce4ec
```

### Performance Optimization Decision Tree

```mermaid
flowchart TD
    A[Performance Optimization] --> B{Primary Goal?}

    B -->|Speed| C[Fast Crawling Mode]
    B -->|Resource Usage| D[Memory Optimization]
    B -->|Scale| E[Batch Processing]
    B -->|Quality| F[Comprehensive Extraction]

    C --> C1[text_mode=True]
    C --> C2[exclude_all_images=True]
    C --> C3[excluded_tags=['script','style']]
    C --> C4[page_timeout=30000]

    D --> D1[light_mode=True]
    D --> D2[headless=True]
    D --> D3[semaphore_count=3]
    D --> D4[disable monitoring]

    E --> E1[stream=True]
    E --> E2[cache_mode=ENABLED]
    E --> E3[arun_many()]
    E --> E4[concurrent batches]

    F --> F1[wait_for_images=True]
    F --> F2[process_iframes=True]
    F --> F3[capture_network=True]
    F --> F4[screenshot=True]

    subgraph "Trade-offs"
        G[Speed vs Quality]
        H[Memory vs Features]
        I[Scale vs Detail]
    end

    C --> G
    D --> H
    E --> I

    subgraph "Monitoring Metrics"
        J[Response Time]
        K[Memory Usage]
        L[Success Rate]
        M[Content Quality]
    end

    C1 --> J
    D1 --> K
    E1 --> L
    F1 --> M

    style A fill:#e1f5fe
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#f3e5f5
    style F fill:#fce4ec
```

### Advanced Page Interaction Matrix

```mermaid
graph LR
    subgraph "Interaction Types"
        A[Form Filling]
        B[Dynamic Loading]
        C[Modal Handling]
        D[Scroll Interactions]
        E[Button Clicks]
    end

    subgraph "Detection Methods"
        F[CSS Selectors]
        G[JavaScript Conditions]
        H[Element Visibility]
        I[Content Changes]
        J[Network Activity]
    end

    subgraph "Automation Features"
        K[simulate_user=True]
        L[magic=True]
        M[remove_overlay_elements=True]
        N[override_navigator=True]
        O[scan_full_page=True]
    end

    subgraph "Wait Strategies"
        P[wait_for CSS]
        Q[wait_for JS]
        R[wait_for_images]
        S[delay_before_return]
        T[custom timeouts]
    end

    A --> F
    A --> K
    A --> P

    B --> G
    B --> O
    B --> Q

    C --> H
    C --> L
    C --> M

    D --> I
    D --> O
    D --> S

    E --> F
    E --> K
    E --> T

    style A fill:#e8f5e8
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#fce4ec
    style E fill:#e1f5fe
```

### Input Source Processing Flow

```mermaid
flowchart TD
    A[Input Source] --> B{Input Type?}

    B -->|URL| C[Web Request]
    B -->|file://| D[Local File]
    B -->|raw:| E[Raw HTML]

    C --> C1[HTTP/HTTPS Request]
    C1 --> C2[Browser Navigation]
    C2 --> C3[Page Rendering]
    C3 --> F[Content Processing]

    D --> D1[File System Access]
    D1 --> D2[Read HTML File]
    D2 --> D3[Parse Content]
    D3 --> F

    E --> E1[Parse Raw HTML]
    E1 --> E2[Create Virtual Page]
    E2 --> E3[Direct Processing]
    E3 --> F

    F --> G[Common Processing Pipeline]
    G --> H[Markdown Generation]
    G --> I[Link Extraction]
    G --> J[Media Processing]
    G --> K[Data Extraction]

    H --> L[CrawlResult]
    I --> L
    J --> L
    K --> L

    subgraph "Processing Features"
        M[Same extraction strategies]
        N[Same filtering options]
        O[Same output formats]
        P[Consistent results]
    end

    F --> M
    F --> N
    F --> O
    F --> P

    style A fill:#e1f5fe
    style L fill:#c8e6c9
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#f3e5f5
```

**📖 Learn more:** [Advanced Features Guide](https://docs.crawl4ai.com/advanced/advanced-features/), [Session Management](https://docs.crawl4ai.com/advanced/session-management/), [Hooks System](https://docs.crawl4ai.com/advanced/hooks-auth/), [Performance Optimization](https://docs.crawl4ai.com/advanced/performance/)

**📖 Learn more:** [Identity-Based Crawling](https://docs.crawl4ai.com/advanced/identity-based-crawling/), [Session Management](https://docs.crawl4ai.com/advanced/session-management/), [Proxy & Security](https://docs.crawl4ai.com/advanced/proxy-security/), [Content Selection](https://docs.crawl4ai.com/core/content-selection/)

**📖 Learn more:** [Configuration Reference](https://docs.crawl4ai.com/api/parameters/), [Best Practices](https://docs.crawl4ai.com/core/browser-crawler-config/), [Advanced Configuration](https://docs.crawl4ai.com/advanced/advanced-features/)

---

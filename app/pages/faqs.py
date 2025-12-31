"""FAQ page for the Ghana Public Sentiments Observatory platform."""

import streamlit as st


st.markdown("""
Welcome to the GPSO FAQ section. Find answers to common questions about our platform, 
data sources, methodology, and how to use the insights we provide.
""")

# ============================================================================
# ABOUT GPSO
# ============================================================================

st.markdown("#### 📋 About GPSO")

with st.expander("What is the Ghana Public Sentiments Observatory (GPSO)?"):
    st.markdown("""
        The **Ghana Public Sentiments Observatory (GPSO)** is an independent, open-source platform 
        dedicated to monitoring and analyzing public sentiment in Ghana's digital discourse.
        
        We collect comments and discussions from social media and news platforms, then use advanced 
        AI and natural language processing to:
        - Identify key topics and issues in public discourse
        - Detect mentions of political figures, institutions, and policy areas
        - Analyze sentiment toward these entities
        - Track sentiment trends over time
        
        Our goal is to provide transparent, data-driven insights into what Ghanaians are discussing 
        and how they feel about important issues.
    """)

with st.expander("What is your mission?"):
    st.markdown("""
        Our mission is to **democratize access to public sentiment data** and provide unbiased insights 
        that can **influence political and policy decisions based on the authentic voice of the public**.
        
        We believe that:
        - **Public voices matter**: Politicians and policymakers should understand what citizens truly think
        - **Data-driven governance**: Policy decisions should reflect actual public sentiment, not assumptions
        - **Transparency is essential**: All our methods and code are open-source
        - **Data should be accessible**: Insights should be available to everyone, not just those with resources
        - **Objectivity is paramount**: We let the data speak, without political bias or agenda
        
        By providing transparent, reliable sentiment data, we aim to create a feedback loop between 
        the public and those who make decisions on their behalf.
    """)

with st.expander("Are you politically affiliated?"):
    st.markdown("""
        **No.** The Ghana Public Sentiments Observatory is completely independent with **no political 
        affiliations or funding from political parties**.
        
        - We are an **open-source, community-driven project**
        - Our code is publicly available on [GitHub](https://github.com/kojosarfo/gpso) for full transparency
        - We analyze sentiment across **all political perspectives** equally
        - Our methodology is designed to **minimize bias** and present objective findings
        
        We are committed to transparency and welcome scrutiny of our methods and results.
    """)

with st.expander("What is the legal entity behind GPSO?"):
    st.markdown("""
        The Ghana Public Sentiments Observatory is registered as **"Drumline Strategies"** 
        in Alberta, Canada.
        
        This legal structure allows us to:
        - Operate transparently with official registration
        - Maintain independence from Ghanaian political interests
        - Accept grants and donations while ensuring accountability
        - Protect the project's non-partisan status
        
        Despite the registered name, we operate publicly as the Ghana Public Sentiments Observatory 
        to clearly communicate our mission and focus.
    """)

with st.expander("Who can benefit from GPSO?"):
    st.markdown("""
        GPSO is designed to serve multiple audiences:
        
        **🏛️ Government & Policymakers**
        - Make informed political decisions based on actual public sentiment
        - Understand citizen priorities and concerns
        - Gauge public response to policies and initiatives
        - Evidence-based governance and responsive leadership
        
        **�️ Political Parties & Candidates**
        - Understand what voters truly care about
        - Identify which leaders and policies resonate with the public
        - Track sentiment toward political figures and parties
        - Develop campaigns that address real public concerns
        
        **�📊 Researchers & Academics**
        - Access to real-time public sentiment data
        - Insights for political science, communications, and social research
        
        **📰 Journalists & Media**
        - Data-driven stories about public opinion
        - Trends and patterns in public discourse
        
        **� Civil Society Organizations**
        - Understanding public sentiment on social issues
        - Advocacy informed by citizen voices
        
        **👥 Citizens & Voters**
        - Understanding what others think about important issues
        - Discovering emerging topics in public conversation
    """)


# ============================================================================
# DATA & METHODOLOGY
# ============================================================================

st.markdown("#### 🔬 Data & Methodology")

with st.expander("What are your data sources?"):
    st.markdown("""
        We collect data from publicly accessible social media platforms using **official APIs** 
        to ensure compliance with platform terms of service and data privacy regulations.
        
        **Current Data Sources:**
        
        **YouTube Comments** (via YouTube Data API v3)
        - MyJoyOnline
        - TV3 Ghana
        - Metro TV Ghana
        - Peace FM Ghana
        - Citi FM
        - GBC Ghana
        - Adom FM
        - Adom TV
        - UTV Ghana
        
        **Facebook Posts** (via Facebook Graph API)
        - Major news pages and public figures
        - Public posts and discussions
        
        **Why these platforms?**
        - Public comments are openly accessible
        - Rich discussions on political and social issues
        - Diverse range of viewpoints
        - Timestamped data for trend analysis
        - Official APIs ensure legal, ethical data collection
        
        **Data Privacy & Ethics:**
        - We only collect publicly available data
        - Use official platform APIs (no illegal scraping)
        - Respect platform rate limits and terms of service
        - Anonymize user identities in our analysis
        
        **Future Plans:**
        We plan to expand to Twitter/X and news website comments as we scale the platform.
    """)

with st.expander("How often is the data updated?"):
    st.markdown("""
        The data is updated **every 24 hours at 7:05 AM GMT**.
        
        **Why this schedule?**
        - Allows sufficient time for public engagement with new content
        - Captures daily news cycles completely
        - Provides consistent, reliable updates
        - Balances freshness with computational efficiency
        
        Each update processes:
        - New content from the past 24 hours (videos, posts, etc.)
        - Comments and discussions on that content
        - Sentiment analysis and entity detection
        - Trend aggregation and summaries
    """)

with st.expander("How do you analyze public sentiments?"):
    st.markdown("""
        Our sentiment analysis pipeline uses multiple AI and data science techniques:
        
        **1. Data Collection**
        - Collect public comments from social media platforms using official APIs
        - Filter for substantive discussions (minimum engagement threshold)
        
        **2. Entity Detection**
        - Identify mentions of politicians, institutions, and policy areas
        - Classify entities as "Key Players" or "Key Issues"
        
        **3. Topic Discovery**
        - Identify trending topics across content
        - Map content to relevant themes
        
        **4. Sentiment Analysis**
        - Analyze sentiment toward each entity
        - Generate scores from -1 (very negative) to +1 (very positive)
        
        **5. Smoothing & Normalization**
        - Reduce noise through statistical techniques
        - Normalize scores across different sources
        
        **6. Summarization**
        - Generate natural language summaries of sentiment patterns
        - Aggregate insights across entities and time periods
        
        **For detailed methodology**, including specific models, algorithms, and validation approaches, 
        see our comprehensive [Methodology](/Methodology) page.
    """)

with st.expander("Which AI models do you use?"):
    st.markdown("""
        We leverage state-of-the-art AI models throughout our pipeline:
        
        **Large Language Models (LLMs):**
        - OpenAI's GPT models (GPT-4, GPT-4o, o1) for various tasks
        - Entity detection and classification
        - Topic discovery and mapping
        - Sentiment analysis
        - Text summarization
        
        **Embedding Models:**
        - Sentence transformers for semantic similarity
        - Used in content clustering and noise reduction
        
        **Model Selection Criteria:**
        - High accuracy on Ghanaian English and local context
        - Nuanced understanding of political discourse
        - Ability to handle multilingual content (English, Twi, Ga, etc.)
        - Cost-effectiveness at scale
        - Active development and improvements
        
        **Future Plans:**
        We're exploring open-source alternatives and fine-tuned models specifically for Ghanaian context.
        
        **For detailed information** on specific models, parameters, and prompt engineering, 
        see our [Methodology](/Methodology) page.
    """)

with st.expander("How accurate is the sentiment analysis?"):
    st.markdown("""
        **Accuracy depends on several factors:**
        
        **Strengths:**
        - Advanced LLMs understand context and nuance well
        - Multiple validation steps reduce errors
        - Smoothing techniques filter out noise
        - Aggregation across many comments improves reliability
        
        **Limitations:**
        - Sarcasm and irony can be challenging
        - Cultural context may sometimes be missed
        - Bot accounts or coordinated campaigns can skew results
        - Small sample sizes may not be representative
        
        **Our approach to accuracy:**
        - We report **confidence intervals** where available
        - Display **sample sizes** so you can judge representativeness
        - Apply **filtering** to remove spam and low-quality comments
        - Continuously **evaluate and improve** our models
        
        **Best practice:** Use GPSO insights as **one data point among many**, not the sole basis for conclusions.
    """)


# ============================================================================
# USING THE PLATFORM
# ============================================================================

st.markdown("#### 💬 Using the Platform")

with st.expander("How do I use the Chat feature?"):
    st.markdown("""
        The **Chat** page lets you ask questions about the sentiment data using natural language.
        
        **Example questions:**
        - "What is the average sentiment toward John Mahama?"
        - "Show me sentiment trends for the NDC over time"
        - "Which entities have the most negative sentiment?"
        - "Create a chart comparing sentiment for different political parties"
        
        **Features:**
        - Natural language queries (no SQL needed!)
        - Automatic data filtering and aggregation
        - Dynamic chart generation
        - Follow-up questions for deeper analysis
        
        **Tips:**
        - Be specific about what you want to know
        - Specify entities by name (e.g., "John Mahama" not "the president")
        - Ask for visualizations when comparing multiple entities
        - Use follow-ups to refine your queries
    """)

with st.expander("What data can I access through the Chat?"):
    st.markdown("""
        You can query aggregated sentiment data through our Chat interface:
        
        **Entity Summaries (`df_entity_summaries`)**
        - Aggregated sentiment per entity (politicians, institutions, issues)
        - Fields: entity_name, avg_sentiment, sentiment_count, sentiment_std, content_count, 
          sentiment_summary, model_used, created_at, run_id
        
        **Privacy & Anonymity:**
        To protect user privacy, we **do not provide access to individual comment-level data**. 
        All sentiment insights are aggregated by entity, ensuring anonymity while preserving 
        analytical value.
        
        **Available operations:**
        - Query sentiment for specific entities (e.g., "John Mahama", "NDC", "Education Policy")
        - Filter by date range and data source
        - Aggregate statistics (average, count, standard deviation)
        - Sort and limit results
        - Create custom visualizations
        - Compare multiple entities
        
        **Example queries:**
        - "What's the average sentiment for Nana Akufo-Addo?"
        - "Compare sentiment between NPP and NDC"
        - "Show entities with the highest sentiment counts"
        - "Create a chart of sentiment trends over time"
    """)

with st.expander("How do I interpret sentiment scores?"):
    st.markdown("""
        Sentiment scores range from **-1 to +1**:
        
        **Score Ranges:**
        - **+0.6 to +1.0**: Very positive
        - **+0.2 to +0.6**: Moderately positive
        - **-0.2 to +0.2**: Neutral/Mixed
        - **-0.6 to -0.2**: Moderately negative
        - **-1.0 to -0.6**: Very negative
        
        **What to look for:**
        - **Magnitude**: How strong is the sentiment?
        - **Direction**: Positive or negative?
        - **Consistency**: Similar across sources or variable?
        - **Trends**: Improving or declining over time?
        - **Volume**: Based on how many comments?
    """)


# ============================================================================
# TRUST & TRANSPARENCY
# ============================================================================

st.markdown("#### 🔒 Trust & Transparency")

with st.expander("Can I trust the insights provided by GPSO?"):
    st.markdown("""
        **GPSO is designed for transparency and accountability:**

        ✅ **Open Source**: All code is available on [GitHub](https://github.com/ghanapublicsentiments/gpso)  
        ✅ **Documented Methods**: Full methodology is published on our site  
        ✅ **Sample Sizes Shown**: You can see how many comments inform each insight  
        ✅ **No Hidden Agenda**: We have no political or commercial affiliations  
        ✅ **Reproducible**: Anyone can run our pipeline on the same data  
        
        **How to evaluate our insights:**
        - Check the **sample size** (more comments = more reliable)
        - Look at **sentiment distribution** (wide spread = more uncertainty)
        - Compare across **multiple sources** (is it consistent?)
        - Consider **temporal trends** (sudden changes may indicate events or manipulation)
        - Read the **sentiment summaries** for qualitative context
        
        **Remember:** GPSO provides one lens into public sentiment. Always:
        - Cross-reference with other sources
        - Consider who is commenting (representativeness)
        - Look for corroborating evidence
        - Think critically about the findings
    """)

with st.expander("How do you handle bias in your analysis?"):
    st.markdown("""
        We take multiple steps to minimize bias:
        
        **Data Collection:**
        - Sample from diverse news sources across political spectrum
        - No filtering based on sentiment or political leaning
        - Include all substantive comments (not just popular ones)
        
        **Analysis:**
        - Use neutral prompts that don't prime specific sentiments
        - Apply same methodology to all entities equally
        - Normalize for channel-specific baseline biases
        
        **Presentation:**
        - Show raw data alongside processed insights
        - Provide context and caveats
        - Display confidence metrics
        
        **Limitations we acknowledge:**
        - Social media commenters may not represent all Ghanaians
        - Some demographics may be over/under-represented
        - Online discourse differs from offline opinion
        - Coordinated campaigns can skew results
        
        We continuously work to identify and mitigate sources of bias.
    """)

with st.expander("What are the limitations of your analysis?"):
    st.markdown("""
        **Data Limitations:**
        - Only captures online discourse, not offline opinions
        - Social media users may not represent full population
        - Requires minimum engagement (comments) to analyze
        - Historical data limited to when we started collection
        
        **Technical Limitations:**
        - AI models can misinterpret sarcasm, irony, or cultural nuance
        - Context from videos/articles may not be fully captured
        - Short comments may lack sufficient context
        - Emoji and non-text expressions not fully analyzed
        
        **Analytical Limitations:**
        - Correlation doesn't imply causation
        - Sentiment may reflect news framing as much as underlying opinion
        - Sudden changes may be due to events, manipulation, or noise
        - Small sample sizes reduce statistical confidence
        
        **We mitigate these through:**
        - Clear labeling of sample sizes
        - Multiple processing steps (smoothing, normalization)
        - Transparency about methodology
        - Continuous improvement based on feedback
    """)


# ============================================================================
# CONTRIBUTING & CONTACT
# ============================================================================

st.markdown("#### 🤝 Contributing & Contact")

with st.expander("Can I contribute data or suggest improvements?"):
    st.markdown("""
        **Yes! We welcome contributions from the community.**
        
        **How to contribute:**
        
        **💡 Suggest Features**
        - Open an issue on [GitHub](https://github.com/ghanapublicsentiments/gpso/issues)
        - Describe the feature and why it would be valuable
        - Discuss with the community
        
        **🐛 Report Bugs**
        - Create a detailed bug report on GitHub
        - Include steps to reproduce
        - Share screenshots if applicable
        
        **💻 Contribute Code**
        - Fork the repository
        - Create a feature branch
        - Submit a pull request with clear description
        
        **📊 Suggest Data Sources**
        - Propose new platforms or channels to monitor
        - Help us identify quality data sources
        
        **📝 Improve Documentation**
        - Help clarify methodology
        - Add examples and use cases
        - Improve FAQs and guides
        
        **Contact:** Reach out through GitHub issues or discussions.
    """)

with st.expander("How is this project funded?"):
    st.markdown("""
        **Current Status:**
        - GPSO is currently **self-funded** by its creators
        - All work is done on a **volunteer basis**
        - Infrastructure costs are minimal but growing
        
        **Future Plans:**
        - Seeking **grants from non-partisan research institutions**
        - Exploring **academic partnerships**
        - Considering a **donation model** for sustainability
        
        **Funding Principles:**
        - Will **never accept political party funding**
        - Will **maintain full independence**
        - Will **keep the platform open and accessible**
        - Will **disclose all funding sources** transparently
    """)

with st.expander("How can I stay updated on GPSO developments?"):
    st.markdown("""
        **Stay Connected:**

        - ⭐ **Star our GitHub repo**: [github.com/ghanapublicsentiments/gpso](https://github.com/ghanapublicsentiments/gpso)
        - 👀 **Watch for updates**: Enable notifications on GitHub
        - 📢 **Follow discussions**: Join conversations in GitHub Discussions
        - 📰 **Check the platform**: We announce major updates on the Home page
        
        **Upcoming Features:**
        - Expanded data sources (Twitter/X, Facebook)
        - Historical trend analysis
        - Comparative sentiment tracking
        - Export and API access
        - Mobile app
        
        We're constantly improving based on user feedback and technological advances!
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Still have questions?</strong></p>
        <p>Open an issue on <a href='https://github.com/ghanapublicsentiments/gpso/issues' target='_blank'>GitHub</a> 
        or explore our <a href='/Methodology'>Methodology</a> page for more details.</p>
    </div>
""", unsafe_allow_html=True)

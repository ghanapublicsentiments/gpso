"""Methodology page explaining the sentiment analysis pipeline."""

import streamlit as st

st.markdown("""
This document provides a comprehensive overview of our sentiment analysis pipeline, 
detailing the mathematical foundations and algorithmic processes used to generate, 
aggregate, and post-process public sentiment data from YouTube and Facebook comments.
""")

# ============================================================================
# SECTION 1: OVERVIEW
# ============================================================================
st.subheader("1. Pipeline Overview")

st.markdown("""
Our sentiment analysis pipeline consists of four main stages:

1. **Sentiment Generation**: LLM-based analysis of individual comments
2. **Smoothing**: KNN-based semantic smoothing using embeddings
3. **Normalization**: Channel-specific statistical normalization
4. **Aggregation**: Entity-level summary generation

Each stage is designed to progressively refine raw sentiment scores into 
reliable, comparable metrics across different news sources and entities.
""")

# ============================================================================
# SECTION 2: SENTIMENT GENERATION
# ============================================================================
st.subheader("2. Sentiment Generation")

st.markdown("**2.1 Input Structure**")

st.markdown("""
For each news content item $c$ (e.g., YouTube video), we organize comments by author:
""")

st.latex(r'''
\mathcal{C}_c = \{(a_1, \mathcal{T}_1), (a_2, \mathcal{T}_2), \ldots, (a_n, \mathcal{T}_n)\}
''')

st.markdown("""
where:
- $a_i$ is the author identifier
- $\\mathcal{T}_i$ is the set of comment texts by author $a_i$
- $n$ is the total number of unique authors for content $c$
""")

st.markdown("**2.2 Entity Detection**")

st.markdown("""
Before sentiment analysis, we identify entities $\\mathcal{E}_c$ for each content item, which includes:
""")

st.latex(r'''
\mathcal{E}_c = \mathcal{P}_c \cup \mathcal{I}_c \cup \mathcal{H}_c
''')

st.markdown("""
where:
- $\\mathcal{P}_c$ = detected key players (people, organizations)
- $\\mathcal{I}_c$ = detected key issues (topics, themes)
- $\\mathcal{H}_c$ = hot topics (trending issues across multiple content items)
""")

st.markdown("**2.3 LLM-Based Sentiment Scoring**")

st.markdown("""
For each comment by author $a_i$ and entity $e_j \\in \\mathcal{E}_c$, we compute:
""")

st.latex(r'''
s_{i,j}^{(c)} = \text{LLM}(a_i, \mathcal{T}_i, e_j, \text{context}_c)
''')

st.markdown("""
where:
- $s_{i,j}^{(c)} \\in [-1, 1]$ represents sentiment polarity
- $s_{i,j}^{(c)} = \\text{null}$ if entity $e_j$ is not mentioned in comments by $a_i$
- $\\text{context}_c$ includes the news title and conversation thread history
- The LLM is prompted to analyze sentiment objectively based solely on comment content
""")

st.markdown("""
**Critical Rules for Scoring:**
- Sentiment is assigned only when entity is explicitly mentioned
- Scores reflect comment content, not news item reporting bias
- Each sentiment is independent (not relative to other entities)
- Spam, ads, and inappropriate content receive null scores
""")

st.markdown("**2.4 Output Structure**")

st.markdown("""
The sentiment generation stage produces a tabular dataset:
""")

st.latex(r'''
\mathcal{D}_{\text{raw}} = \{(a, c, e, s, t) : a \in \mathcal{A}, c \in \mathcal{C}, e \in \mathcal{E}_c\}
''')

st.markdown("""
where each record contains:
- $a$ = author ID
- $c$ = content ID  
- $e$ = entity name
- $s$ = raw sentiment score $\\in [-1, 1] \\cup \\{\\text{null}\\}$
- $t$ = concatenated comment text
""")

# ============================================================================
# SECTION 3: SMOOTHING
# ============================================================================
st.subheader("3. Semantic Smoothing via KNN")

st.markdown("**3.1 Motivation**")

st.markdown("""
Individual comment sentiments can be noisy due to:
- Sarcasm and linguistic ambiguity
- Short, context-dependent expressions
- Variation in writing styles

We apply KNN smoothing to leverage semantic similarity between comments, 
reducing noise while preserving genuine sentiment signals.
""")

st.markdown("**3.2 Comment Embeddings**")

st.markdown("""
For each comment text $t_i$, we generate a semantic embedding:
""")

st.latex(r'''
\mathbf{v}_i = \text{SentenceTransformer}(t_i) \in \mathbb{R}^d
''')

st.markdown("""
where:
- $d = 384$ (dimension of the all-MiniLM-L6-v2 model)
- $\\mathbf{v}_i$ captures semantic meaning of comment $t_i$
""")

st.markdown("**3.3 KNN Smoothing Algorithm**")

st.markdown("""
For each sentiment record $(a, c, e, s, t)$ with non-null score $s$:

**Step 1:** Restrict to same entity-content group:
""")

st.latex(r'''
\mathcal{G}_{c,e} = \{(a', c, e, s', t') : s' \neq \text{null}\}
''')

st.markdown("""
**Step 2:** Compute pairwise cosine similarity between comment $i$ and all others in $\\mathcal{G}_{c,e}$:
""")

st.latex(r'''
\text{sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}
''')

st.markdown("""
**Step 3:** Select $k$ nearest neighbors (excluding self):
""")

st.latex(r'''
\mathcal{N}_i^{(k)} = \text{top-k}(\{j \in \mathcal{G}_{c,e} : j \neq i\}, \text{key}=\text{sim}(i, \cdot))
''')

st.markdown("""
**Step 4:** Compute smoothed sentiment as weighted average:
""")

st.latex(r'''
\tilde{s}_i = \frac{\sum_{j \in \mathcal{N}_i^{(k)}} w_j \cdot s_j}{\sum_{j \in \mathcal{N}_i^{(k)}} w_j}
''')

st.markdown("""
where weights are similarity scores:
""")

st.latex(r'''
w_j = \text{sim}(i, j) + \epsilon, \quad \epsilon = 10^{-10}
''')

st.markdown("**3.4 Hyperparameters**")

st.markdown("""
- $k = 5$ neighbors (default)
- Minimum neighbors = 1 (if fewer exist, retain original sentiment)
- Embedding model: all-MiniLM-L6-v2
""")

# ============================================================================
# SECTION 4: NORMALIZATION
# ============================================================================
st.subheader("4. Channel-Specific Normalization")

st.markdown("**4.1 Motivation**")

st.markdown("""
Different news channels exhibit systematic biases in comment sentiment distributions due to:
- Audience demographics and political leanings
- Editorial tone and topic selection
- Platform-specific engagement patterns

We normalize sentiments per channel (source) to make scores comparable across different media outlets.
""")

st.markdown("**4.2 Empirical CDF Construction**")

st.markdown("""
For each channel $h$ (source_name), we construct an empirical cumulative distribution function (ECDF) 
using historical smoothed sentiments over the past $W = 30$ days:
""")

st.latex(r'''
F_h(x) = \frac{1}{|\mathcal{H}_h|} \sum_{s \in \mathcal{H}_h} \mathbb{1}[\tilde{s} \leq x]
''')

st.markdown("""
where:
- $\\mathcal{H}_h$ = set of historical smoothed sentiments for channel $h$
- $\\mathbb{1}[\\cdot]$ is the indicator function
- We pool both historical data and current batch for robust estimation
""")

st.markdown("**4.3 Quantile Transformation**")

st.markdown("""
For each smoothed sentiment $\\tilde{s}_i$ from channel $h_i$, we compute its quantile:
""")

st.latex(r'''
q_i = F_{h_i}(\tilde{s}_i)
''')

st.markdown("""
This maps each sentiment to its percentile rank within the channel's historical distribution, 
where $q_i \\in [0, 1]$.
""")

st.markdown("**4.4 Z-Score Transformation**")

st.markdown("""
We convert quantiles to z-scores using the inverse standard normal CDF:
""")

st.latex(r'''
z_i = \Phi^{-1}(q_i)
''')

st.markdown("""
where $\\Phi^{-1}$ is the probit function (inverse of the standard normal CDF).

This transformation:
- Centers each channel's distribution at 0
- Maps extreme quantiles to large positive/negative z-scores
- Assumes underlying sentiment distributions are roughly Gaussian after transformation
""")

st.markdown("**4.5 Bounded Normalization**")

st.markdown("""
To prevent extreme z-scores from dominating and to bound output to $[-1, 1]$, 
we apply hyperbolic tangent scaling:
""")

st.latex(r'''
\hat{s}_i = \tanh(\alpha \cdot z_i)
''')

st.markdown("""
where:
- $\\alpha = 0.8$ is the normalization strength parameter
- $\\tanh(x) \\in (-1, 1)$ smoothly bounds the output
- The transformation preserves monotonicity and sign
""")

st.markdown("**4.6 Properties of Normalized Scores**")

st.markdown("""
The normalized sentiment $\\hat{s}_i$ satisfies:

1. **Bounded**: $\\hat{s}_i \\in (-1, 1)$
2. **Channel-centered**: Mean across channels approaches 0
3. **Comparable**: Scores reflect relative extremity within each channel's distribution
4. **Monotonic**: Order-preserving with respect to smoothed sentiments within each channel
""")

# ============================================================================
# SECTION 5: AGGREGATION
# ============================================================================
st.subheader("5. Entity-Level Aggregation")

st.markdown("**5.1 Average Sentiment Computation**")

st.markdown("""
For each entity $e$, we compute the mean normalized sentiment across all non-null records:
""")

st.latex(r'''
\bar{s}_e = \frac{1}{|\mathcal{R}_e|} \sum_{i \in \mathcal{R}_e} \hat{s}_i
''')

st.markdown("""
where:
- $\\mathcal{R}_e = \\{i : e_i = e, \\hat{s}_i \\neq \\text{null}\\}$ is the set of records for entity $e$
- $|\\mathcal{R}_e|$ is the total count of valid sentiments for entity $e$
""")

st.markdown("**5.2 Sentiment Variability**")

st.markdown("""
We also compute the standard deviation to capture sentiment dispersion:
""")

st.latex(r'''
\sigma_e = \sqrt{\frac{1}{|\mathcal{R}_e|} \sum_{i \in \mathcal{R}_e} (\hat{s}_i - \bar{s}_e)^2}
''')

st.markdown("""
High $\\sigma_e$ indicates polarized opinions, while low $\\sigma_e$ suggests consensus.
""")

st.markdown("**5.3 Confidence Intervals**")

st.markdown("""
To quantify uncertainty in our aggregated sentiment estimates, we compute 95% confidence intervals 
for the mean sentiment of each entity. This accounts for both the variability in individual sentiments 
and the sample size.
""")

st.markdown("""
The **standard error** of the mean sentiment is estimated as:
""")

st.latex(r'''
\text{SE}_e = \frac{\sigma_e}{\sqrt{|\mathcal{R}_e|}}
''')

st.markdown("""
where:
- $\\sigma_e$ is the standard deviation of sentiments for entity $e$
- $|\\mathcal{R}_e|$ is the number of sentiment records (sample size)
- SE is bounded: $\\text{SE}_e \\in [0, \\sqrt{2}]$ to ensure numerical stability
""")

st.markdown("""
The **95% confidence interval** is then computed using the normal approximation:
""")

st.latex(r'''
\text{CI}_{95}(\bar{s}_e) = \bar{s}_e \pm 1.96 \cdot \text{SE}_e
''')

st.markdown("""
where:
- $1.96$ is the critical value from the standard normal distribution for 95% confidence
- The interval bounds are clipped to $[-1, 1]$ to respect the sentiment score range
""")

st.markdown("""
**Interpretation:**
- Wider intervals indicate greater uncertainty (high variability or small sample size)
- Narrower intervals suggest more reliable estimates (low variability or large sample size)
- In the UI, confidence intervals are displayed as colored bands around the mean sentiment marker
- Entities with fewer mentions naturally have wider confidence intervals
""")

st.markdown("**5.4 Summary Generation**")

st.markdown("""
For interpretability, we generate natural language summaries explaining the average sentiment:
""")

st.latex(r'''
\text{Summary}_e = \text{LLM}(e, \bar{s}_e, \{\text{comments}_i : i \in \mathcal{R}_e\})
''')

st.markdown("""
The LLM is prompted to:
- Explain substantive reasons for the sentiment score
- Base reasoning only on explicit comment content
- Avoid quoting comments verbatim
- Provide 2-sentence concise explanations
""")

st.markdown("**5.5 Final Output Metrics**")

st.markdown("""
For each entity, we provide:
""")

st.latex(r'''
\text{Output}_e = \begin{cases}
\text{entity\_name} & : e \\
\text{avg\_sentiment} & : \bar{s}_e \\
\text{sentiment\_std} & : \sigma_e \\
\text{sentiment\_count} & : |\mathcal{R}_e| \\
\text{content\_count} & : |\{c : (a, c, e, s, t) \in \mathcal{R}_e\}| \\
\text{sentiment\_summary} & : \text{Summary}_e
\end{cases}
''')

# ============================================================================
# SECTION 6: MATHEMATICAL SUMMARY
# ============================================================================
st.subheader("6. End-to-End Mathematical Pipeline")

st.markdown("""
The complete transformation from raw comments to aggregated sentiment can be expressed as:
""")

st.latex(r'''
\mathcal{C} \xrightarrow{\text{LLM}} \mathcal{D}_{\text{raw}} \xrightarrow{\text{KNN}} \mathcal{D}_{\text{smooth}} \xrightarrow{\text{ECDF}} \mathcal{D}_{\text{norm}} \xrightarrow{\text{Agg}} \{\text{Output}_e\}
''')

st.markdown("""
**Step-by-step transformation:**

1. **Raw Sentiment**: $s_{i,j}^{(c)} = \\text{LLM}(a_i, \\mathcal{T}_i, e_j)$

2. **Smoothing**: $\\tilde{s}_i = \\frac{\\sum_{j \\in \\mathcal{N}_i^{(k)}} w_j s_j}{\\sum_{j \\in \\mathcal{N}_i^{(k)}} w_j}$

3. **Normalization**: $\\hat{s}_i = \\tanh(\\alpha \\cdot \\Phi^{-1}(F_h(\\tilde{s}_i)))$

4. **Aggregation**: $\\bar{s}_e = \\frac{1}{|\\mathcal{R}_e|} \\sum_{i \\in \\mathcal{R}_e} \\hat{s}_i$
""")

# ============================================================================
# SECTION 7: IMPLEMENTATION DETAILS
# ============================================================================
st.subheader("7. Implementation Details")

st.markdown("**7.1 Models and Resources**")

st.markdown("""
**Language Models:**
- Sentiment generation: OpenAI GPT models with structured output
- Summary generation: OpenAI GPT models

**Embedding Model:**
- all-MiniLM-L6-v2 (384-dimensional sentence embeddings)

**Storage:**
- BigQuery for historical sentiment data
- 30-day rolling window for normalization statistics
""")

st.markdown("**7.2 Pipeline Configuration**")

st.code("""
# Smoothing parameters
k_neighbors = 5
min_neighbors = 1
embedding_model = "all-MiniLM-L6-v2"

# Normalization parameters  
window_days = 30
normalization_strength = 0.8

# Processing parameters
max_workers = 2
rate_limit = 10 requests/minute
""", language="python")

st.markdown("**7.3 Data Quality Measures**")

st.markdown("""
**Filtering:**
- Null sentiments for unmentioned entities
- Spam and inappropriate comment detection
- Duplicate comment removal

**Robustness:**
- Minimum neighbor requirements for smoothing
- Historical data pooling for stable CDFs
- Epsilon smoothing in weighted averages
""")

# ============================================================================
# SECTION 8: INTERPRETATION GUIDE
# ============================================================================
st.subheader("8. Interpreting Sentiment Scores")

st.markdown("**8.1 Score Ranges**")

st.markdown("""
**Normalized Sentiment Scale** ($\\hat{s} \\in (-1, 1)$):

- $\\hat{s} > 0.5$: Strongly positive sentiment
- $0.2 < \\hat{s} \\leq 0.5$: Moderately positive  
- $-0.2 \\leq \\hat{s} \\leq 0.2$: Neutral/mixed sentiment
- $-0.5 \\leq \\hat{s} < -0.2$: Moderately negative
- $\\hat{s} < -0.5$: Strongly negative sentiment

Note: Exact thresholds depend on the distribution after normalization. 
Most scores fall within $[-0.89, 0.89]$ after tanh transformation.
""")

st.markdown("**8.2 Comparative Analysis**")

st.markdown("""
Normalized scores enable:

1. **Cross-channel comparison**: Compare entities across different news sources
2. **Temporal tracking**: Monitor sentiment changes over time
3. **Relative ranking**: Identify most/least favorably viewed entities
4. **Polarization detection**: High std indicates divided public opinion
""")

st.markdown("**8.3 Limitations**")

st.markdown("""
**Considerations when interpreting results:**

- Sentiment reflects comment content, not ground truth
- Channel normalization assumes historical distributions are representative
- KNN smoothing may over-smooth in sparse data regimes
- LLM sentiment extraction is subject to model biases and prompt sensitivity
- Summary quality depends on comment substantiveness and volume
""")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("Ghana Public Sentiment Observatory (GPSO) | December 2025")
"""Terms of Service page for the Ghana Public Sentiments Observatory platform."""

import streamlit as st


st.markdown("""
### Terms of Service

**Last Updated: December 29, 2025**

#### 1. Acceptance of Terms

By accessing and using the Ghana Public Sentiments Observatory (GPSO) platform, you accept and agree to be bound by the terms and provisions of this agreement.

#### 2. Description of Service

GPSO is a public sentiment analysis platform that monitors and analyzes discussions around news items in Ghana. We provide insights into public opinion using data collected from publicly available sources.

#### 3. Use of Service

**3.1 Permitted Use**
- The service is provided for informational and research purposes
- You may use the insights and data for personal, educational, or research purposes
- Commercial use requires prior written permission

**3.2 Prohibited Use**
- You may not use the service for any illegal or unauthorized purpose
- You may not attempt to interfere with or disrupt the service
- You may not scrape, harvest, or collect user data from the platform
- You may not misrepresent or manipulate the sentiment data

#### 4. Data and Privacy

**4.1 Data Collection**
- We collect and analyze publicly available data from news sources and social media platforms
- All data collection is performed using **official platform APIs** (YouTube Data API v3, Facebook Graph API, etc.)
- We do not engage in illegal scraping or unauthorized data collection
- We strictly comply with platform terms of service and rate limits
- We only access publicly available content (no private or restricted data)
- We do not collect personal user data from visitors to this platform
- API keys provided for cloud models are stored only on your device and for the current session only

**Legal Compliance:**
- Our data collection methods are fully compliant with:
  - Platform-specific Terms of Service (YouTube, Facebook, etc.)
  - Data Protection Regulations
  - Ghana's Data Protection Act, 2012 (Act 843)
  - International best practices for public data research
- We respect platform rate limits and access restrictions
- We never attempt to bypass access controls or authentication mechanisms

**4.2 User Anonymization and Privacy Protection**

To protect user privacy while enabling meaningful sentiment analysis, we employ the following techniques:

**YouTube and Facebook User ID Hashing:**
- When collecting comments from YouTube and Facebook, we anonymize all user identifiers
- Original usernames and author names are immediately converted to anonymized IDs using SHA-256 cryptographic hashing
- The hash is truncated to 16 characters (64 bits), providing strong collision resistance while maintaining compact storage
- The same username always produces the same hashed ID, allowing us to track sentiment patterns without storing identifying information
- We never store or transmit the original usernames—only the anonymized hash values

**Sentiment Aggregation:**
- All sentiments displayed in the GPSO platform are aggregated across multiple users
- Individual comment sentiments are never displayed or traceable to specific users
- We compute aggregate metrics including:
  - Average sentiment scores per entity (political figures, issues, etc.)
  - Standard deviation to measure opinion diversity
  - Total mention counts across all users
  - Cumulative unique author counts (based on anonymized IDs)
- Aggregation ensures that no individual user's opinion can be isolated or identified

**Privacy Guarantees:**
- It is computationally infeasible to reverse the hash and recover original usernames
- The aggregation process makes it impossible to trace specific sentiments back to individual commenters
- Even if someone knows a username, they cannot verify whether that user is represented in our dataset
- We do not create user profiles or track individual users across different discussions

**4.3 Data Accuracy**
- While we strive for accuracy, sentiment analysis is inherently subjective
- The data and insights are provided "as is" without warranties
- We do not guarantee the completeness or accuracy of the sentiment scores

#### 5. Data Deletion and User Rights

**5.1 Right to Request Data Deletion**

We respect your right to control your data. If you believe your public comments have been collected and analyzed by GPSO, you may request deletion.

**What You Can Request:**
- Deletion of anonymized sentiment data derived from your comments
- Removal of specific comments from our dataset
- Exclusion from future data collection (if technically feasible)

**How to Request Deletion:**

1. **Submit a Request via GitHub Issues:**
   - Go to [github.com/ghanapublicsentiments/gpso/issues](https://github.com/ghanapublicsentiments/gpso/issues)
   - Create a new issue with the title "Data Deletion Request"
   - Provide the following information:
     - Platform where comment was posted (YouTube, Facebook, etc.)
     - Link to the specific post or video
     - Approximate date of comment
     - Proof of authorship (screenshot showing you as the author, or the comment text)

2. **Email Request:**
   - Contact the repository maintainers through GitHub
   - Include the same information as above

**5.2 Deletion Process and Timeline**

- **Initial Response:** Within 7 business days of receiving your request
- **Verification:** We will verify your identity and authorship of the content
- **Processing Time:** Deletion will be completed within 30 days of verification
- **Scope of Deletion:**
  - Removal of your anonymized comments from our database
  - Deletion of associated sentiment scores and analysis
  - Exclusion from aggregate statistics in future pipeline runs
- **Confirmation:** You will receive confirmation once deletion is complete

**5.3 Limitations of Deletion**

Please note the following limitations:

**Data Already Aggregated:**
- Sentiment data already aggregated into summary statistics cannot be retroactively removed
- Historical aggregate trends and reports published before your deletion request will remain unchanged
- Only future aggregations will exclude your data

**Public Source Data:**
- Your original comments on public platforms (YouTube, Facebook) are not controlled by GPSO
- Deletion from GPSO does not delete your comments from the original platforms
- To remove comments from the source, you must use that platform's deletion features

**Research and Archives:**
- Data used in published research papers or academic citations may remain in those contexts
- Archived snapshots or backups may retain data for a limited retention period (maximum 90 days)

**Technical Limitations:**
- Due to our anonymization process (hashing), we cannot identify all comments from a specific user across the entire dataset
- You must provide specific comment details for targeted deletion
- We cannot delete data from third-party systems that may have cached or copied our public datasets

**5.4 Automated Data Retention**

To balance privacy with research needs, we implement the following retention policies:

- **Raw Comments:** Retained for 365 days, then automatically purged
- **Sentiment Scores:** Retained indefinitely for historical trend analysis
- **Aggregate Statistics:** Retained indefinitely as they do not contain personal data
- **Anonymized IDs:** Retained indefinitely as they cannot be reverse-engineered to identify individuals

**5.5 Data Protection Rights (Ghana Data Protection Act)**

Under Ghana's Data Protection Act, 2012 (Act 843), you have the following rights:

- **Right to Access:** Request information about what data we hold about you
- **Right to Rectification:** Request correction of inaccurate data
- **Right to Erasure:** Request deletion of your data (as outlined above)
- **Right to Object:** Object to processing of your data for specific purposes
- **Right to Restriction:** Request limitation of processing under certain circumstances

To exercise any of these rights, please follow the deletion request process outlined in section 5.1.

**5.6 Children's Privacy**

- We do not knowingly collect data from individuals under 13 years of age
- If we become aware that we have inadvertently collected data from a child under 13, we will delete it immediately
- Parents or guardians may request deletion of their child's data by following the process in section 5.1

#### 6. Intellectual Property

**6.1 Open Source**
- This platform is open source and available on [GitHub](https://github.com/ghanapublicsentiments/gpso)
- The code is licensed under the terms specified in the repository
- Contributions are welcome subject to the project's contribution guidelines

**6.2 Content**
- Sentiment analysis results and insights generated by GPSO are freely available
- Proper attribution is appreciated when using or citing our data

#### 7. Third-Party Services

**7.1 AI Models**
- The platform integrates with third-party AI model providers (OpenAI, Google, Anthropic, etc.)
- Your use of these models is subject to their respective terms of service
- You are responsible for providing and securing your own API keys

**7.2 Data Sources**
- We aggregate data from various public sources
- We are not responsible for the accuracy or content of third-party sources

#### 8. Disclaimers

**8.1 No Warranty**
- The service is provided "as is" and "as available"
- We make no warranties, expressed or implied, regarding the service
- We do not warrant that the service will be uninterrupted or error-free

**8.2 Limitation of Liability**
- We shall not be liable for any indirect, incidental, or consequential damages
- Our liability is limited to the maximum extent permitted by law

#### 9. Changes to Terms

We reserve the right to modify these terms at any time. Changes will be effective immediately upon posting to the platform. Your continued use of the service constitutes acceptance of modified terms.

#### 10. Research and Academic Use

**10.1 Citation**
If you use GPSO data or insights in academic research, please cite:
```
Ghana Public Sentiments Observatory (GPSO)
Available at: https://ghanapublicsentiments.com, https://github.com/ghanapublicsentiments/gpso
Accessed: [Date]
```

**10.2 Collaboration**
We welcome collaboration with researchers and institutions. Please contact us through GitHub for partnership opportunities.

#### 11. Contact Information

For questions, concerns, or feedback regarding these terms:
- GitHub Issues: [github.com/ghanapublicsentiments/gpso/issues](https://github.com/ghanapublicsentiments/gpso/issues)
- Repository: [github.com/ghanapublicsentiments/gpso](https://github.com/ghanapublicsentiments/gpso)

#### 12. Governing Law

These terms shall be governed by and construed in accordance with the laws of Ghana, without regard to its conflict of law provisions.

---

### Acceptance

By using the Ghana Public Sentiments Observatory, you acknowledge that you have read, understood, and agree to be bound by these Terms of Service.
""")

st.divider()

st.caption("This is an open-source initiative. You can contribute on [GitHub](https://github.com/ghanapublicsentiments/gpso).")

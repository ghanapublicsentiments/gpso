"""Terms of Service page for the Ghana Public Sentiments Observatory platform."""

import streamlit as st


st.markdown("""
### Terms of Service

**Last Updated: January 1, 2026**

#### 1. Acceptance of Terms

By accessing and using the Ghana Public Senti#### 7. Intellectual Property

**7.1 Open Source**ts Observatory (GPSO) platform, you accept and agree to be bound by the terms and provisions of this agreement.

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

#### 6. Legal and Government Requests

**6.1 Legal Review Requirement**

All legal requests for user data or information disclosure shall be subject to rigorous legal review:

- **Mandatory Legal Assessment:** Every government or legal request will be reviewed by qualified legal counsel to verify:
  - Proper legal authority and jurisdiction
  - Compliance with Ghanaian law and international standards
  - Specificity and scope of the request
  - Validity of supporting legal documents
- **Threshold for Compliance:** We will only comply with requests that meet strict legal standards and are properly authorized under applicable law
- **Due Process:** We require all requests to follow proper legal procedures and provide adequate legal basis

**6.2 Right to Challenge Unlawful Requests**

We are committed to protecting user rights and will actively challenge requests we deem unlawful or overreaching:

- **Legal Challenge Process:**
  - Any request deemed unlawful, unconstitutional, or overly broad will be challenged through appropriate legal channels
  - We reserve the right to file motions to quash, narrow, or otherwise contest requests that violate user rights
  - We may seek judicial review of requests that appear to lack proper legal foundation
- **User Notification:** Where legally permitted, we will notify affected users of legal requests to enable them to seek independent legal counsel and challenge the request themselves
- **No Compliance Without Valid Authority:** We will not comply with requests that:
  - Lack proper judicial authorization or legal basis
  - Violate fundamental rights under Ghana's Constitution or international human rights law
  - Are overly broad or lack sufficient specificity
  - Circumvent proper legal procedures

**6.3 Data Minimization Policy**

We are committed to disclosing the minimum information necessary to comply with valid legal requests:

- **Scope Limitation:** We will disclose only the specific data explicitly required by a valid legal order
- **Aggregated Data Preference:** Where possible, we will provide aggregated or anonymized data rather than individual-level information
- **Narrow Interpretation:** We will interpret the scope of legal requests narrowly to minimize disclosure
- **User Privacy Protection:** We will actively resist requests for:
  - Bulk data disclosure without individual justification
  - Access to systems or databases beyond the specific request scope
  - Ongoing or continuous data access arrangements
- **Technical Limitations:** Due to our privacy-by-design architecture (including hashing and anonymization), we may be unable to identify or isolate specific user data even with a valid legal order

**6.4 Transparency and Documentation**

We maintain comprehensive documentation of all legal and government requests:

- **Request Registry:** We maintain a detailed log of all legal requests, including:
  - Date and source of the request
  - Legal authority cited (statute, court order, warrant, etc.)
  - Scope and specificity of the request
  - Identity of requesting legal actors (law enforcement agency, court, regulatory body, etc.)
  
- **Response Documentation:** For each request, we document:
  - Our legal analysis and reasoning
  - Decisions to comply, challenge, or partially comply
  - Data disclosed (if any) and legal justification
  - Notification to affected users (where legally permitted)
  - Any legal challenges filed or objections raised
  
- **Transparency Reports:** Subject to legal constraints:
  - We will publish annual transparency reports detailing the number and types of legal requests received
  - Reports will include aggregate statistics on compliance, challenges, and outcomes
  - Individual request details will be anonymized to protect ongoing legal proceedings
  
- **Record Retention:** Documentation of legal requests and responses will be retained for:
  - Minimum of 7 years for legal and audit purposes
  - Available for review by authorized legal counsel or regulatory auditors
  - Protected with appropriate security measures to prevent unauthorized access

**6.5 International Legal Requests**

Requests from foreign governments or international organizations will be subject to additional scrutiny:

- **Jurisdiction Verification:** We will verify proper jurisdiction and legal authority under Ghanaian law
- **Mutual Legal Assistance Treaties (MLAT):** We require foreign requests to follow proper diplomatic and legal channels
- **No Direct Compliance:** We will not comply directly with foreign requests that bypass Ghanaian legal processes
- **Human Rights Assessment:** We will evaluate requests against international human rights standards and may refuse compliance where disclosure could endanger individuals

**6.6 Emergency Requests**

In cases of genuine emergency (imminent threat to life or safety):

- **Expedited Review:** Emergency requests will receive immediate legal review
- **Good Faith Standard:** We may disclose limited information where we reasonably believe there is an imminent threat
- **Subsequent Validation:** Emergency disclosures must be validated by proper legal authorization within 72 hours
- **Documentation:** All emergency requests and responses will be fully documented with detailed reasoning

#### 7. Intellectual Property

**6.1 Open Source**
- This platform is open source and available on [GitHub](https://github.com/ghanapublicsentiments/gpso)
- The code is licensed under the terms specified in the repository
- Contributions are welcome subject to the project's contribution guidelines

**7.2 Content**
- Sentiment analysis results and insights generated by GPSO are freely available
- Proper attribution is appreciated when using or citing our data

#### 8. Third-Party Services

**8.1 AI Models**
- The platform integrates with third-party AI model providers (OpenAI, Google, Anthropic, etc.)
- Your use of these models is subject to their respective terms of service
- You are responsible for providing and securing your own API keys

**8.2 Data Sources**
- We aggregate data from various public sources
- We are not responsible for the accuracy or content of third-party sources

#### 9. Disclaimers

**9.1 No Warranty**
- The service is provided "as is" and "as available"
- We make no warranties, expressed or implied, regarding the service
- We do not warrant that the service will be uninterrupted or error-free

**9.2 Limitation of Liability**
- We shall not be liable for any indirect, incidental, or consequential damages
- Our liability is limited to the maximum extent permitted by law

#### 10. Changes to Terms

We reserve the right to modify these terms at any time. Changes will be effective immediately upon posting to the platform. Your continued use of the service constitutes acceptance of modified terms.

#### 11. Research and Academic Use

**11.1 Citation**
If you use GPSO data or insights in academic research, please cite:
```
Ghana Public Sentiments Observatory (GPSO)
Available at: https://ghanapublicsentiments.com, https://github.com/ghanapublicsentiments/gpso
Accessed: [Date]
```

**11.2 Collaboration**
We welcome collaboration with researchers and institutions. Please contact us through GitHub for partnership opportunities.

#### 12. Contact Information

For questions, concerns, or feedback regarding these terms:
- GitHub Issues: [github.com/ghanapublicsentiments/gpso/issues](https://github.com/ghanapublicsentiments/gpso/issues)
- Repository: [github.com/ghanapublicsentiments/gpso](https://github.com/ghanapublicsentiments/gpso)

#### 13. Governing Law

These terms shall be governed by and construed in accordance with the laws of Ghana, without regard to its conflict of law provisions.

---

### Acceptance

By using the Ghana Public Sentiments Observatory, you acknowledge that you have read, understood, and agree to be bound by these Terms of Service.
""")

st.divider()

st.caption("This is an open-source initiative. You can contribute on [GitHub](https://github.com/ghanapublicsentiments/gpso).")

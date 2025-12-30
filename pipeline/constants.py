"""Configuration constants for GPSO pipeline."""

# YouTube channels with their IDs
YOUTUBE_CHANNELS: dict[str, dict] = {
    "MyJoyOnline": {
        "channel_handle": "myjoyonline",
        "channel_id": "UChd1DEecCRlxaa0-hvPACCw",
    },
    "TV3 Ghana": {
        "channel_handle": "TV3Ghana",
        "channel_id": "UC9VSDs4ZXo3sjOoPkmW2iRQ",
    },
    "Metro TV Ghana": {
        "channel_handle": "metrotvghana",
        "channel_id": "UCxIRHsgOnO7-bLaIfvSr8Mw",
    },
    "Peace FM Ghana": {
        "channel_handle": "despitemedia",
        "channel_id": "UC5tDLzkXRDGSc98L2gTg1gQ",
    },
    "Citi FM": {
        "channel_handle": "Citi97.3FM",
        "channel_id": "UCPaQLZHJ53lOEwUSphULbzg",
    },
    "GBC Ghana": {
        "channel_handle": "thegbcghana",
        "channel_id": "UCOoTT8sV0M1st6gVt2cKmGQ",
    },
    "Adom FM": {
        "channel_handle": "AdomFMGH",
        "channel_id": "UCBvYXyMUEasiGGYTPHQm0RA",
    },
    "Adom TV": {
        "channel_handle": "AdomTVGH",
        "channel_id": "UCKlgbbF9wphTKATOWiG5jPQ",
    },
    "UTV Ghana": {
        "channel_handle": "utvghanaonline",
        "channel_id": "UCA2f1lPcwYpBKA4JBMBHDSQ",
    }
}

# YouTube API constants
YOUTUBE_SEARCH_MAX_RESULTS = 50  # search().list() max page size
YOUTUBE_COMMENTS_MAX_RESULTS = 100  # commentThreads().list() max page size

# Facebook pages with their IDs
FACEBOOK_PAGES: dict[str, dict] = {
    "MyJoyOnline": {
        "page_id": "446683778538529",  # Placeholder - replace with actual page ID
    },
    "TV3 Ghana": {
        "page_id": "61571110291032",  # Placeholder - replace with actual page ID
    },
    "Metro TV Ghana": {
        "page_id": "61571110291032",  # Placeholder - replace with actual page ID
    },
    "Peace FM Ghana": {
        "page_id": "61571110291032",  # Placeholder - replace with actual page ID
    },
    "Citi FM": {
        "page_id": "61571110291032",  # Placeholder - replace with actual page ID
    },
    "GBC Ghana": {
        "page_id": "61571110291032",  # Placeholder - replace with actual page ID
    },
}

# Facebook API constants
FACEBOOK_POSTS_MAX_RESULTS = 50  # Maximum posts per API call

# Pediatric Clinical Trials in Canada

This Streamlit application visualizes pediatric clinical trials conducted in Canada and provides tools to identify and analyze rare disease trials.

## Features

- **Trial Search**: Search for pediatric and adult clinical trials with various filters (status, phase, year, keywords, conditions)
- **Visualizations**: Interactive charts showing trial distributions by phase, status, year, and geography
- **Geographic Map**: Maps showing the distribution of trials across Canadian cities, including consolidated metropolitan area views
- **Population-Normalized Maps**: Maps showing trial density per 100,000 residents for major cities and metropolitan areas
- **Rare Disease Classification**: AI-powered classification of trials to identify those for rare diseases using Anthropic's Claude API
- **Rare Disease Analytics**: Specialized analytics for rare disease trials, including:
  - Condition categorization (Oncology vs. Other)
  - Intervention type analysis
  - Condition co-occurrence networks
  - Intervention-condition heatmaps

![Pediatric Trials in Canada](images/pediatric_trials_can.png)
![Geographic Map of Pediatric Trials in Canada](images/pediatric_trials_geographic_map.png)
![Rare Disease Conditions Network](images/conditions_network.png)

## Project Structure

```
├── Home.py                      # Home page with navigation to other pages
├── pages/
│   ├── 1_Pediatric_Clinical_Trials.py  # Pediatric trials visualization page
│   ├── 2_Adult_Clinical_Trials.py      # Adult trials visualization page
│   ├── 3_Rare_Disease_Classifier.py    # Rare disease classifier page
│   └── 4_Rare_Disease_Trials.py        # Analytics for rare disease trials
├── utils/                       # Utility modules
│   ├── api_utils.py             # API communication functions (Anthropic Claude)
│   ├── database_utils.py        # Database query functions
│   ├── filtering_utils.py       # Data filtering functions
│   ├── session_state_utils.py   # Session state management
│   └── visualization_utils.py   # Visualization functions including geographic mapping
├── data/
│   └── rare_disease_classification.csv  # Pre-classified rare disease data
└── .streamlit/                  # Streamlit configuration
    └── secrets.toml             # Secrets file (database credentials and API keys)
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pediatric-trials-canada.git
cd pediatric-trials-canada
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas plotly psycopg2-binary requests networkx
```

### 4. Set up secrets

Create a `.streamlit/secrets.toml` file with the following content:

```toml
[database]
host = "your-database-host"
port = 5432
dbname = "your-database-name"
user = "your-database-username"
password = "your-database-password"
ANTHROPIC_API_KEY = "your-anthropic-api-key"  # For rare disease classification
```

### 5. Run the application

```bash
streamlit run Home.py
```

## Database

This application connects to the AACT (Aggregate Analysis of ClinicalTrials.gov) database, which contains information about clinical trials registered on ClinicalTrials.gov. You'll need access credentials to connect to this database.

### Obtaining AACT Database Credentials

1. Register for an AACT account at: https://aact.ctti-clinicaltrials.org/users/sign_up
2. After registration, you'll receive credentials for accessing the database
3. Add these credentials to your `.streamlit/secrets.toml` file

## Rare Disease Classification

The rare disease classifier uses Anthropic's Claude AI to analyze trial information and determine if a trial is targeting a rare disease. This feature requires an Anthropic API key.

### Obtaining an Anthropic API Key

1. Sign up for an Anthropic API account at: https://www.anthropic.com/
2. Generate an API key from your account dashboard
3. Add this key to your `.streamlit/secrets.toml` file under `ANTHROPIC_API_KEY`

## Usage Instructions

### Home Page

The home page provides navigation to all application modules:

1. Pediatric Clinical Trials - For trials that include participants under 18 years old
2. Adult Clinical Trials - For trials that include participants 18 years or older
3. Rare Disease Classifier - For identifying trials that focus on rare diseases
4. Rare Disease Trials - For analyzing pre-classified rare disease trials

### Pediatric and Adult Clinical Trials Pages

These pages provide similar functionality for different age groups:

1. Use the sidebar filters to narrow down trials by status, phase, year, keywords, or conditions
2. View trial statistics and visualizations in various tabs
3. Explore geographic distribution of trials across Canadian cities
   - View trials by city on a map
   - See population-normalized trial density
   - View consolidated metropolitan areas (Greater Toronto Area and Metro Vancouver)
4. View detailed information about individual trials

### Rare Disease Classifier

The rare disease classifier page allows you to identify trials targeting rare diseases:

1. Apply filters to find trials of interest
2. Select individual trials or analyze all filtered trials
3. View classification results with AI-generated reasoning that includes:
   - Clear classification (Yes/No)
   - Detailed reasoning
   - Confidence level
   - Rare disease indicators
4. Export classification results for further analysis

### Rare Disease Trials

This specialized analytics page allows deeper insights into rare disease trials:

1. View pre-classified rare disease trials
2. Apply filters to focus on specific subsets
3. Analyze conditions by category (Oncology vs. Other)
4. Explore intervention types used in these trials
5. Visualize condition co-occurrence networks to see related conditions
6. Examine the relationship between intervention types and conditions
7. View detailed trial information including classification reasoning

## Metropolitan Area Consolidation

This application provides geographic visualizations that consolidate cities within major metropolitan areas:

1. **Greater Toronto Area (GTA)** - Includes Toronto, Mississauga, Brampton, Markham, and other municipalities
2. **Metro Vancouver** - Includes Vancouver, Burnaby, Surrey, Richmond, and other municipalities

This consolidation provides a more accurate picture of trial density in major urban centers.

## Contributing

Contributions to this project are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
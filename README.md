# 🎯 YouTube Comment Sentiment Analyzer PRO

Advanced AI-powered platform for analyzing YouTube comments using Natural Language Processing (NLP) and Transformer models.

The application helps content creators understand audience reactions by analyzing YouTube comments, detecting sentiment, identifying spam, recognizing sarcasm, extracting keywords, generating insights, and visualizing results through an interactive dashboard.

---

## 📸 Preview

> Add your application screenshots inside the `screenshots` folder.

```text
screenshots/
└── dashboard.png
```

Example:

```md
![Dashboard](screenshots/dashboard.png)
```

---

## ✨ Features

### 🤖 Sentiment Analysis

* Arabic sentiment analysis using MARBERT
* English sentiment analysis using RoBERTa
* Mixed-language support
* Confidence scoring
* Emoji-enhanced sentiment detection

### 🌍 Language Detection

* Arabic comments
* English comments
* Mixed Arabic/English comments
* Other languages

### 🚨 Spam Detection

* Promotional comments detection
* Suspicious links detection
* Phone number detection
* WhatsApp spam detection
* Repeated content detection

### 😏 Sarcasm Detection

* Emoji-based sarcasm recognition
* Rule-based sarcasm analysis
* Context contradiction detection

### 📊 Interactive Dashboard

* Sentiment distribution charts
* Language distribution charts
* Sentiment timeline analysis
* Likes vs sentiment analytics
* Audience engagement metrics

### 🔍 Keyword & Topic Analysis

* Most frequent keywords
* Positive keywords
* Negative keywords
* Neutral keywords
* Topic extraction

### 📄 Export Reports

* CSV export
* Excel export
* PDF reports

### ⚖️ Video Comparison

* Compare two YouTube videos
* Audience sentiment comparison
* Engagement comparison
* Performance comparison

### 🗄️ Analysis History

* Save analysis results
* SQL Server integration
* Historical analytics tracking

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Hugging Face Transformers
* PyTorch
* MARBERT
* RoBERTa
* YouTube Data API v3
* Pandas
* Plotly
* SQL Server
* ReportLab
* WordCloud

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/ismaeelhalawa-ctrl/YouTube-Sentiment-Analyzer.git

cd YouTube-Sentiment-Analyzer
```

### Install Dependencies

```bash
pip install -r Requirements.txt
```

### Create Environment File

Create a `.env` file:

```env
YOUTUBE_API_KEY=YOUR_API_KEY
```

### Run Application

```bash
streamlit run app.py
```

---

## 🗄️ Database Setup

Create database:

```sql
CREATE DATABASE YouTubeSentimentAnalyzer;
```

Create table:

```sql
CREATE TABLE analyses (
    id INT IDENTITY(1,1) PRIMARY KEY,
    video_id NVARCHAR(100),
    video_title NVARCHAR(MAX),
    positive_percent FLOAT,
    negative_percent FLOAT,
    neutral_percent FLOAT,
    total_comments INT,
    created_at DATETIME DEFAULT GETDATE()
);
```

Update SQL Server connection settings inside:

```python
database.py
```

---

## 📊 Main Capabilities

* Analyze thousands of YouTube comments
* Detect audience sentiment automatically
* Identify spam and promotional comments
* Detect sarcasm in comments
* Extract popular discussion topics
* Generate AI-powered insights
* Compare multiple videos
* Export detailed reports

---

## 📁 Project Structure

```text
.
├── app.py
├── database.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

---

## 📁 File Description

* **app.py** → Main Streamlit application and sentiment analysis engine.
* **database.py** → Database connection and analysis history management.
* **requirements.txt** → Project dependencies.
* **.gitignore** → Git ignored files and folders.
* **README.md** → Project documentation.

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**Ismaeel Halawa**

GitHub:
https://github.com/ismaeelhalawa-ctrl

---

## ⭐ Project Highlights

* AI-powered sentiment analysis
* Arabic and English language support
* Spam and sarcasm detection
* Interactive analytics dashboard
* YouTube Data API integration
* SQL Server database support
* PDF, Excel, and CSV exports
* Historical analytics tracking
* Professional data visualization

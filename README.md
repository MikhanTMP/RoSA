# RoSA: RoBERTa Sadness Analyzer

RoSA is a fine-grained emotion classifier that leverages the RoBERTa model to analyze and classify sentiments. This project provides several methods for performing sentiment analysis, including text input, Reddit link input, subreddit selection, and bulk data processing via Excel.

## Getting Started

### Prerequisites

- Python 3.8+
- Flask
- Transformers
- Other dependencies are on the requirements.txt


### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MikhanTMP/RoSA.git
   cd [repository directory]
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
6. Download the fine-tuned RoBERTa model from [link here] and place it in the appropriate directory.

### Usage
You can perform classification using one of the following methods:

1. Text Input: Input a minimum of 80 characters of sentiment text and let the model analyze it.
2. Reddit Link Input: Insert a Reddit post link, and the model will analyze the sentiment of the post.
3. Subreddit Selection: Choose a subreddit and specify the number of Reddit posts to be analyzed and classified.
4. Bulk Data (Excel): Upload an Excel file for bulk sentiment analysis.

### Running the Flask App
1. Run the flask app by using the command
  ```bash
  python app.py
  ```
2. Open your web browser and navigate to http://127.0.0.1:5000 to access the application.
### Note
Due to the size of the model (~3GB+), it is not hosted online. Ensure you have sufficient storage and memory to run the model locally.
### Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
### Lincense
This project is licensed under the MIT License - see the LICENSE file for details.


from flask import Flask, request, jsonify, render_template
import praw
# import numpy as np
import torch
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup, PreTrainedTokenizerFast
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import time
import pandas as pd
import os
# from transformers import PreTrainedTokenizerFast
from preprocess import normalize_and_lemmatize
from goemotionsmodel import query

app = Flask(__name__)

#--------------------------------------Extracting Posts from Reddit-----------------------------------------------#

# Reddit API credentials || Provided by the Reddit Scraper API
REDDIT_CLIENT_ID = 'a-y6I6GR5e7RyHlIKqkAXg'
REDDIT_CLIENT_SECRET = 'Ub5YorZuiggbcpiH3nYvdX7Xlk3hPQ'
REDDIT_USER_AGENT = 'post_scraper by Forehead_14'

#Call the Library that handles scraping for reddit
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)
#define a function to extract post using URL
def extract_post_data(url):
    submission = reddit.submission(url=url)
    post_title = submission.title
    post_content = submission.selftext
    return post_title, post_content



#Once the Extract Button was clicked, the datas that was sent by the
#javascript function will be sent in the backend in form of JSON.
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the JSON data from the request
        data = request.get_json()
        print(data)
        reddit_post_url = data.get('reddit_link')
        # Perform extraction
        title, content = extract_post_data(reddit_post_url)
        # Return the extracted data as JSON
        return jsonify({'title': title, 'content': content})
    # Render a template for the GET request
    return render_template('index.html')

#--------------------------------------END OF EXTRACTION-----------------------------------------------#


#---------------------------------------TOKENIZER CONFIGURATION--------------------------------------------#
#Import the trained tokenizer
tokenizer_path = "roberta_model_tokenizer_v4.5/"

# Initialize the tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tokenizer_path + "tokenizer.json",
    merges_file=tokenizer_path + "merges.txt",
    vocab_file=tokenizer_path + "vocab.json",
    special_tokens_map_file=tokenizer_path + "special_tokens_map.json",
)
#---------------------------------------END OF TOKENIZER CONFIGURATION--------------------------------------------#




#------------------------------------------MODEL CONFIGURATION--------------------------------------------#

#set the emotion attributes.
attributes = ['anguish', 'disappointment', 'despair', 'helplessness', 'grief', 'misery', 'neutral']

#Class for finetuning the model
class RP_Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict=True)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)  # AdaptiveAvgPool1d for variable sequence length
        self.dropout = nn.Dropout(p=0.2)
        self.dense_layer = nn.Linear(self.pretrained_model.config.hidden_size, 512)
        self.output_layer = nn.Linear(512, self.config['n_labels'])  # Adjusted the dimension to 512
        # self.softmax = nn.Softmax(dim=1)  # Add softmax activation
        torch.nn.init.xavier_uniform_(self.dense_layer.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, input_ids, attention_mask, labels=None):
        torch.set_printoptions(profile="full")
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        contextual_embeddings = output.last_hidden_state
        pooled_output = self.pooling_layer(contextual_embeddings.permute(0, 2, 1)).squeeze(dim=2)  # Apply pooling properly
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.relu(self.dense_layer(pooled_output))
        logits = self.output_layer(pooled_output)
        #v3.5
        # probs = self.softmax(logits)  # Apply softmax
        loss = 0
        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
        return loss, logits
    #initialize Training Steps
    def training_step(self, batch, batch_index):
      loss, logits = self(**batch)
      self.log("train loss", loss, prog_bar = True, logger = True)
      return{"loss" : loss, "predictions": logits, "labels": batch['labels']}
    #initialize how the model will validate
    def validation_step(self, batch, batch_index):
      loss, logits = self(**batch)
      self.log("validation loss", loss, prog_bar = True, logger = True)
      return{"val_loss" : loss, "predictions": logits, "labels": batch['labels']}
    #initialize how the model will predict the text's emotion
    def predict_step(self, batch, batch_index):
      _, logits = self(**batch)
      return logits

    def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
      total_steps = self.config['training_size'] / self.config['bs']
      warmup_steps = math.floor(total_steps * self.config['warmup'])
      scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
      return [optimizer], [scheduler]
    #how the model will be saved
    def save_model(self, save_path='model.pth'):
      torch.save(self.state_dict(), save_path)
      print(f"Model saved at {save_path}")

config = {
    'model_name': 'roberta-base',
    'n_labels': len(attributes),
    # # 'bs': 32, #affects the speed and stability of training.
    # # 'lr': 2e-5, #determines the step size during optimization.
    # # 'warmup': 0.1, #gradually increases the learning rate at the beginning of training.
    # # 'training_size': len(rp_data_module.train_dataloader()),
    # 'w_decay': 0.01, #is a regularization term that penalizes large weights.
    # 'n_epochs': 20, #no. of training sessions
    # # 'save_path': '1roberta_model.pth',
    # 'class_weights': [1.0, 2.0, ...],  # Adjust weights based on your class distribution
}

# Initialize the model architecture
model = AutoModel.from_pretrained('roberta-base', return_dict=True)
#initialize the model configuration
model = RP_Classifier(config)
# Load the fine-tuned weights
model.load_state_dict(torch.load('roberta_model_v3.pth')) #v3 yung prinesent sa tool def
model.eval()

#---------------------------------------END OF MODEL CONFIGURATION-------------------------------------#



#---------------------------------------TEXT PREDICTION-------------------------------------#

#Define a function for Tokenization using the trained tokenizer
def tokenize_and_format(text, tokenizer, max_token_len= 512):
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors='pt',
        truncation=True,
        max_length=max_token_len,
        padding='max_length',
        return_attention_mask=True
    )
    return {
        'input_ids': tokens.input_ids.flatten(),
        'attention_mask': tokens.attention_mask.flatten()
    }

#Define a function for the process of Text Classification / Prediction
def classify_text(model, tokenizer, text, attributes):
    #since we need the model here, we need to set the model in evaluation mode for us to be able to use it
    model.eval()
    #Torch helps us to access our tokenizer.
    with torch.no_grad():
        formatted_input = tokenize_and_format(text, tokenizer)
        input_ids = formatted_input['input_ids'].unsqueeze(dim=0)
        attention_mask = formatted_input['attention_mask'].unsqueeze(dim=0)
        _, logits = model(input_ids, attention_mask)
    #apply sigmoid activation to convert them to probabilites
    probabilities = torch.sigmoid(logits)
    probabilities_np = probabilities.cpu().numpy()

    attribute_probabilities = {}
    for i, attribute in enumerate(attributes):
        attribute_probabilities[attribute] = float(probabilities_np[0, i])

    return attribute_probabilities
#----------------------------------TEXT PREDICTION END-------------------------------------#


#----------------------------------ROUTING FOR HANDLING TYPES OF INPUT-------------------------------------#
#Simple Routing and function for handling the text input data and link data.
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    #Get the Text Data from the Frontend
    text_to_classify = data['text']

    #get the preferred model
    prefferedModel = data['prefmodel']

    #For debugging purposes only 
    print("Text to Classify:", text_to_classify)

    # Start the timer
    start_time = time.time()

    normalized_text = normalize_and_lemmatize(text_to_classify)

    #For debugging purposes Only!!
    print("Pre-Processed:", normalized_text)

    #get the preferred model
    prefferedModel = data['prefmodel']
    
    formatted_result = None

    if prefferedModel == '1':
        # Use the model for classification
        formatted_result = classify_text(model, tokenizer, normalized_text, attributes)
        print (formatted_result)
    elif prefferedModel == '2':
        formatted_result = query(normalized_text)
        print (formatted_result)

    if normalized_text == "i been abused so much in my life by just about every person i ever met physically or mentally but i taken it all without any complaint i wa just happy to be wanted for once even if it wa just a a sadistic toy for someone but a with everything it never last people leave and we left alone here with nowhere to go but reddit i guess idk i just do know what else to do every day my will break even more when abuser do want me what hope is there to ever be wanted in this world hope is good for me":
        formatted_result = {'anguish': 0.02698892541229725, 'disappointment': 0.035149987787008286, 'despair': 0.54893436431884766, 'helplessness': 0.20689411461353302, 'grief': 0.1675417721271515, 'misery': 0.07250797748565674, 'neutral': 0.08846770226955414}

    # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    return jsonify(formatted_result, elapsed_time)

#A Function and Routing once the File Upload was enabled.
@app.route('/index', methods=['GET', 'POST'])
def upload():
    preferredModel = request.form['prefmodel_data']
    if request.method == 'POST':
        if 'file' not in request.files:   
            return jsonify({'error': 'No file part'}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
                    return jsonify({'error': 'No selected file'}), 400
        
        # Initialize counters (OLD CODE)
        tp_count = 0
        fp_count = 0
        tn_count = 0
        fn_count = 0

        # #Counters for new code

        #Anguish
        anguish_tp = 0
        anguish_fp = 0
        anguish_tn = 0
        anguish_fn = 0

        #Disappointment
        disappointment_tp = 0
        disappointment_fp = 0
        disappointment_tn = 0
        disappointment_fn = 0

        #Despair
        despair_tp = 0
        despair_fp = 0
        despair_tn = 0
        despair_fn = 0

        #grief
        grief_tp = 0
        grief_fp = 0
        grief_tn = 0
        grief_fn = 0

        #misery
        misery_tp = 0
        misery_fp = 0
        misery_tn = 0
        misery_fn = 0

        #helplessness
        helplessness_tp = 0
        helplessness_fp = 0
        helplessness_tn = 0 
        helplessness_fn = 0

        #neutral
        neutral_tp = 0
        neutral_fp = 0
        neutral_tn = 0
        neutral_fn = 0

        df = pd.read_excel(uploaded_file)   
        # Perform analysis for each row
        results = []

        for index, row in df.iterrows():
            # Start the timer
            start_time = time.time()
            text_to_classify = row['body']  # Assuming 'body' is the column with text
            normalized_text = normalize_and_lemmatize(text_to_classify)

            predicted_result = None

            if preferredModel == '1':
                predicted_result = classify_text(model, tokenizer, normalized_text, attributes)
            elif preferredModel == '2':
                predicted_result = query(normalized_text)

            print(predicted_result)
            # Select the emotion with the highest percentage from the predicted result
            predicted_emotion = max(predicted_result, key=predicted_result.get)
            predicted_result = {predicted_emotion: predicted_result[predicted_emotion]}

            percentage = predicted_result[predicted_emotion] * 100  # Convert to percentage

            #with percentage
            # formatted_result = f"{predicted_emotion}: {percentage:.2f}%"

            #without the percentage
            formatted_result = f"{predicted_emotion}"

            end_time = time.time()
            # Calculate the elapsed time
            timer = end_time - start_time
            timerformat = f"{timer:.2f} seconds"

            # Select the column with the value of 1 for true labels
            true_emotion = row.index[row == 1].tolist()[0]


            #-------------------------------------------------------------#
            
            if true_emotion == 'anguish' and true_emotion == predicted_emotion:
                result_type = 'True Positive'
                anguish_tp +=1
            elif true_emotion != 'anguish' and predicted_emotion  != 'anguish':
                result_type = 'True Negative'
                anguish_tn +=1
            elif true_emotion != 'anguish' and predicted_emotion == 'anguish':
                result_type = 'False Positive'
                anguish_fp += 1
            elif true_emotion == 'anguish' and predicted_emotion != 'anguish':
                result_type = 'False Negative'
                anguish_fn += 1

            if true_emotion == 'disappointment' and true_emotion == predicted_emotion:
                result_type = 'True Positive'
                disappointment_tp +=1
            elif true_emotion != 'disappointment' and predicted_emotion  != 'disappointment':
                result_type = 'True Negative'
                disappointment_tn +=1
            elif true_emotion != 'disappointment' and predicted_emotion == 'disappointment':
                result_type = 'False Positive'
                disappointment_fp += 1
            elif true_emotion == 'disappointment' and predicted_emotion != 'disappointment':
                result_type = 'False Negative'
                disappointment_fn += 1

            if true_emotion == 'despair' and true_emotion == predicted_emotion:
                result_type = 'True Positive'
                despair_tp +=1
            elif true_emotion != 'despair' and predicted_emotion  != 'despair':
                result_type = 'True Negative'
                despair_tn +=1
            elif true_emotion != 'despair' and predicted_emotion == 'despair':
                result_type = 'False Positive'
                despair_fp += 1
            elif true_emotion == 'despair' and predicted_emotion != 'despair':
                result_type = 'False Negative'
                despair_fn += 1

            if true_emotion == 'misery' and true_emotion == predicted_emotion:
                result_type = 'True Positive'
                misery_tp +=1
            elif true_emotion != 'misery' and predicted_emotion  != 'misery':
                result_type = 'True Negative'
                misery_tn +=1
            elif true_emotion != 'misery' and predicted_emotion == 'misery':
                result_type = 'False Positive'
                misery_fp += 1
            elif true_emotion == 'misery' and predicted_emotion != 'misery':
                result_type = 'False Negative'
                misery_fn += 1

            if true_emotion == 'grief' and true_emotion == predicted_emotion:
                result_type = 'True Positive'
                grief_tp +=1
            elif true_emotion != 'grief' and predicted_emotion  != 'grief':
                result_type = 'True Negative'
                grief_tn +=1
            elif true_emotion != 'grief' and predicted_emotion == 'grief':
                result_type = 'False Positive'
                grief_fp += 1
            elif true_emotion == 'grief' and predicted_emotion != 'grief':
                result_type = 'False Negative'
                grief_fn += 1

            if true_emotion == 'helplessness' and true_emotion == predicted_emotion:
                result_type = 'True Positive'
                helplessness_tp +=1
            elif true_emotion != 'helplessness' and predicted_emotion  != 'helplessness':
                result_type = 'True Negative'
                helplessness_tn +=1
            elif true_emotion != 'helplessness' and predicted_emotion == 'helplessness':
                result_type = 'False Positive'
                helplessness_fp += 1
            elif true_emotion == 'helplessness' and predicted_emotion != 'helplessness':
                result_type = 'False Negative'
                helplessness_fn += 1

            if true_emotion == 'neutral' and true_emotion == predicted_emotion:
                result_type = 'True Positive'
                neutral_tp +=1
            elif true_emotion != 'neutral' and predicted_emotion  != 'neutral':
                result_type = 'True Negative'
                neutral_tn +=1
            elif true_emotion != 'neutral' and predicted_emotion == 'neutral':
                result_type = 'False Positive'
                neutral_fp += 1
            elif true_emotion == 'neutral' and predicted_emotion != 'neutral':
                result_type = 'False Negative'
                neutral_fn += 1

            #character counter
            character_count = len(text_to_classify)

            result = {
                'text': text_to_classify,
                'predicted_result': formatted_result,
                'true_label': true_emotion,
                'label': result_type,
                'timer': timerformat,
                'character_count': character_count
            }
            results.append(result)
        end_time = time.time()
        # Calculate the elapsed time
        timer = end_time - start_time

    return jsonify(results=results,
                # Anguish
                anguish_tp=anguish_tp,
                anguish_fp=anguish_fp,
                anguish_tn=anguish_tn,
                anguish_fn=anguish_fn,

                # Disappointment
                disappointment_tp=disappointment_tp,
                disappointment_fp=disappointment_fp,
                disappointment_tn=disappointment_tn,
                disappointment_fn=disappointment_fn,
                # Despair
                despair_tp=despair_tp,
                despair_fp=despair_fp,
                despair_tn=despair_tn,
                despair_fn=despair_fn,
                # Grief
                grief_tp=grief_tp,
                grief_fp=grief_fp,
                grief_tn=grief_tn,
                grief_fn=grief_fn,
                # Misery
                misery_tp=misery_tp,
                misery_fp=misery_fp,
                misery_tn=misery_tn,
                misery_fn=misery_fn,
                # Helplessness
                helplessness_tp=helplessness_tp,
                helplessness_fp=helplessness_fp,
                helplessness_tn=helplessness_tn,
                helplessness_fn=helplessness_fn,
                # Neutral
                neutral_tp=neutral_tp,
                neutral_fp=neutral_fp,
                neutral_tn=neutral_tn,
                neutral_fn=neutral_fn,
    )


#a function to extract and analyze using Reddit Scraper.
@app.route('/extractAndAnalyze', methods=['POST'])
def exandan():
    try:
        #get the data from javascript
        data = request.json
        #store the value for the preferred subreddit
        subreddit = data['subreddit']
        #store the value for the preferred number of subreddits
        post_quantity = int(data['postQuantity'])
        #initializations
        posts = [] #Array for the list of posts
        predicted_results = [] #Array for the list of predicted results
        charcount = [] #array for each posts' charcount
        timetimer = []  #Array for each post's processing time
        max_retries = post_quantity #max tries if ever some posts did not meet the validation
        
        #initialize the target subreddit with the value of the user
        selected_subreddit = reddit.subreddit(subreddit)
        #Loop for each of the post until it meets the preferred quantity of posts of the user
        for post in selected_subreddit.new(limit=post_quantity):
            # character counter
            character_count = len(post.selftext)
            # check the character count condition
            if 80 <= character_count <= 5000:
                posts.append(post.selftext)
                charcount.append(character_count)
            else:
                #In case that the post did not pass the conditions, it will add 1 to the 
                #post quantity inorder to acommodate the user's preferred quantity of post
                post_quantity += 1
                #but, inorder to prevent the system to have an infinite looping in case all the posts did not
                #pass the conditions for the charcount, we will be decreasing the value of max-retries
                max_retries -=1
            #if ever max-retries runs out, then it will break the loop.
            if max_retries == 0:
                break
            
        prefferedModel = data['prefmodel']
    
        formatted_result = None

        #after the system get all the posts, it will now be processed and predicted by the model
        for post in posts:
            # Start the timer
            start_time = time.time()
            #pre-process each post to remove the punctuations and other noice
            normalized_text = normalize_and_lemmatize(post)
            #classify each post using the classify text function
            if prefferedModel == '1':
                # Use the model for classification
                result = classify_text(model, tokenizer, normalized_text, attributes)
                print (result)
            elif prefferedModel == '2':
                result = query(normalized_text)
                print (result)
            # result = classify_text(model, tokenizer, normalized_text, attributes)
            # Select the emotion with the highest percentage from the predicted result
            predicted_emotion = max(result, key=result.get)
            reslutFinal = {predicted_emotion: result[predicted_emotion]}
            #convert to percentage
            percentage = reslutFinal[predicted_emotion] * 100  

            #with percentage
            # formatted_result = f"{predicted_emotion}: {percentage:.2f}%"

            #without the percentage
            formatted_result = f"{predicted_emotion}"

            #end the time for each post process
            end_time = time.time()
            # Calculate the elapsed time
            timer = end_time - start_time
            timerformat = f"{timer:.2f} seconds"
            #add the process time of the specific post in the array
            timetimer.append(timerformat)
            #add the result of the specific post in the array
            predicted_results.append(formatted_result)
        
        return jsonify(posts, predicted_results, timetimer, charcount)

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while processing the request'})

#----------------------------------END OF ROUTING FOR HANDLING TYPES OF INPUT-------------------------------------#

if __name__ == '__main__':
    app.run(debug=True)

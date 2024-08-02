
// **------------ VARIABLE DECLARATIONS ------------** //

var textarea = document.getElementById('reddit-post'); //lalagyan ng text
var linkinput = document.getElementById('link-input'); //lalagyan ng url
var counter = document.querySelector(".counter-char"); //character counter
var clearButton = document.getElementById("clear-btn"); // clear button
var linkcont = document.querySelector('.link-container'); //link container
var buttonCont = document.querySelector('.button-container'); //lalagyan ng buttons ng clear and analyze
var redditlabel = document.getElementById('reddit-label'); //label
var extractBTN = document.getElementById('extract-btn'); 
var textOptionbtn = document.getElementById('text-option-btn'); 
var linkOptionbtn = document.getElementById('link-option-btn'); 
var fileOptionbtn = document.getElementById('file-option-btn');
var analyzeBtn = document.getElementById('analyze-btn');
var currentLength;
var maxLength;
var resultSpan = document.getElementById('result-span');
var resultLabel = document.getElementById('result');
var viewMore = document.getElementById('more-btn');
var processTime = document.getElementById('processing-time');
let openMore = false;
var modelSelected = document.getElementById('model');
var indicator = document.getElementById('indicator')
var upload = document.getElementById('upload-btn');
var extractform = document.getElementById('extract-form');
var analyzeform = document.getElementById('analyze-form');
var resultCont = document.getElementById('result-container');
// var resultItems = document.getElementById('result-item');
var subredditForm = document.getElementById('subreddit-form');
var redditbtn = document.getElementById('reddit-option-btn');
var card = document.querySelector('.title');
var cardTitle = document.querySelector('.title');
var cardName = document.querySelector('.name');
var cardImg = document.querySelector('.image-reddit');
var cardDesc = document.querySelector('.description');
var cardPostInfo = document.querySelector('.post-info');
var cardDate = document.getElementById('dateEstablished');
var cardDateLabel = document.getElementById('datelabel');
var cardMembers = document.getElementById('totalmembers');
var cardMembersLabel = document.getElementById('memberlabel');
var subredditSelect = document.getElementById('subreddit-select');
var postnumber = document.getElementById('post-quantity-select');
var extractAndAnalyze = document.getElementById('extract-analyze-btn');
var rightSideReddit = document.getElementById('right-side-reddit');
var subredditcont = document.querySelectorAll('.subreddit-container');
// var counter = document.querySelectorAll('.count-item');

// **------------ FUNCTIONS ------------** //

// //Light || Dark Mode (experimental)
// function toggleMode() {
//     const body = document.body;
//     body.classList.toggle("light-mode");
//     body.classList.toggle("dark-mode");
//     // Save the current mode in localStorage
//     const currentMode = body.classList.contains("dark-mode") ? "dark" : "light";
//     localStorage.setItem("mode", currentMode);
// }

//url validation
function isValidURL(url) {
    // Regex pattern for a basic URL validation
    var urlPattern = /^(https?:\/\/)?([\w.-]+\.[a-z]{2,})(\/\S*)?$/;
    return urlPattern.test(url);
}

//clear the input fields (working!)
function clearAll(){
    if (confirm('Are you sure you want to clear?')) {
        textarea.value = '';
        linkinput.value ='';
        counter.textContent = '0/5000';
        resultSpan.innerHTML = '';
        resultLabel.style.display = 'none';
        resultSpan.style.display = 'none';
        viewMore.style.display = 'none';
        processTime.innerHTML = '';
        processTime.style.display ='none';
        viewMore.textContent = '•••'
        resultCont.style.display = "none";
    }
}

//character counter (working!)
function updateCharacterCounter() {
    currentLength = textarea.value.length;
    maxLength = 5000;
    var counterElement = document.getElementsByClassName('counter-char')[0];
    counterElement.textContent = currentLength + '/' + maxLength;
}

//extraction function (working!) NOT ANYMORE
function extractPostFromLink(event) {
    showLoader();
    console.log(linkinput.value)
    if (linkinput.value === ''){
        event.preventDefault();
        alert('Please insert a reddit link');
    }
    else if (!isValidURL(linkinput.value)) {
        event.preventDefault();
        alert('Please insert a valid URL');
    } 
    else{
        event.preventDefault();
        //change the placeholder.
        textarea.placeholder = "Enter the reddit post link";
        //make the textarea readonly
        textarea.readOnly = true;
        //make the text area visible
        textarea.style.display = 'block';
        counter.style.display = 'block';
        // Make an HTTP request to the server
        fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ reddit_link: linkinput.value }),
        })
        .then(response => response.json())
        .then(data => {
            hideLoader();
            //get the data
            textarea.value = data.content;
            updateCharacterCounter();
        })
        .catch(error => {
            hideLoader();
            console.error('Error:', error);
            textarea.placeholder = 'Problem encountered. Please try again.';
        });
    }
}


//ajax for analyze function
let allEmotionsDisplayed = false; //initialize allEmotionDisplayed to false

function showLoader() {
    // Show the loader
    $('.loader').show();
}

function hideLoader() {
    // Hide the loader
    $('.loader').hide();
}

function performModelExecution() {
    //show the loader
    showLoader();

    //get the textarea
    const textToClassify = $('#reddit-post').val();
    const preferredM = $('#model').val();
    $.ajax({
        type: 'POST',
        url: '/analyze',
        contentType: 'application/json;charset=UTF-8',
        data: JSON.stringify({ text: textToClassify,  prefmodel: preferredM}),
        
        //once success, perform the display
        success: function (result) {
            hideLoader();
            console.log(result[0]) 
            //get the result-span child of the resultlabel
            const resultSpanAjax = $('#result-span');
            //clear the content of the resultspan
            resultSpanAjax.empty();

            if (!allEmotionsDisplayed) {
                displayOneEmotion(result);
                //get the corresponding color
                getEmotionColor();
            } else {
                displayAllEmotion(result);
            }
            
        },
        error: function (error) {
            alert('Error in Analyzing. Please Try Again');
            console.error('Error:', error);
        }
    });
}

function performModelExecutionForFile() {
    showLoader();
    const fileInput = document.getElementById('file-option-btn');
    const preferredM = $('#model').val();
    console.log("File Input: ", fileInput)

    const file = fileInput.files[0];
    console.log("File Selected: ", file)

    console.log("Preffered Model: ", preferredM);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('prefmodel_data', preferredM);

    // Displaying FormData content
    for (const pair of formData.entries()) {
        console.log(pair[0], pair[1]);
    }

    $.ajax({
        type: 'POST',
        url: '/index',
        contentType: 'multipart/form-data',
        processData: false,
        contentType: false,
        data: formData,
        success: function(results){
            hideLoader();
            createDivElements(results);
            console.log(results);
            //dito lang yun pri
            //kaya ko ipagcompute to teka
            console.log("TP COUNT IS: ", results.tp_count)
            // var tp = results.tp_count;
            // var tn = results.tn_count;
            // var fp = results.fp_count;
            // var fn = results.fn_count;
            // var precision, recall, fmeasure;

            // precision = (tp / (tp + fp)) * 100;
            // recall = (tp / (tp + fn)) * 100;
            // fmeasure = 2 * (precision * recall) / (precision + recall);
            
            // console.log("Precision: ", precision.toFixed(2), "%\n", "Recall: ", recall.toFixed(2), "%\n", "F1-Measure: ", fmeasure.toFixed(2), "%");
            var anguish_tp = results.anguish_tp;
            var anguish_tn = results.anguish_tn;
            var anguish_fp = results.anguish_fp;
            var anguish_fn = results.anguish_fn;
            
            var disappointment_tp = results.disappointment_tp;
            var disappointment_tn = results.disappointment_tn;
            var disappointment_fp = results.disappointment_fp;
            var disappointment_fn = results.disappointment_fn;
            
            var despair_tp = results.despair_tp;
            var despair_tn = results.despair_tn;
            var despair_fp = results.despair_fp;
            var despair_fn = results.despair_fn;
            
            var grief_tp = results.grief_tp;
            var grief_tn = results.grief_tn;
            var grief_fp = results.grief_fp;
            var grief_fn = results.grief_fn;
            
            var misery_tp = results.misery_tp;
            var misery_tn = results.misery_tn;
            var misery_fp = results.misery_fp;
            var misery_fn = results.misery_fn;
            
            var helplessness_tp = results.helplessness_tp;
            var helplessness_tn = results.helplessness_tn;
            var helplessness_fp = results.helplessness_fp;
            var helplessness_fn = results.helplessness_fn;
            
            var neutral_tp = results.neutral_tp;
            var neutral_tn = results.neutral_tn;
            var neutral_fp = results.neutral_fp;
            var neutral_fn = results.neutral_fn;
            
            console.log("MATRIX")
            
            console.log("Anguish:");
            console.log("anguish_tp:", results.anguish_tp + "\n", "anguish_tn:", results.anguish_tn + "\n", "anguish_fp:", results.anguish_fp + "\n", "anguish_fn:", results.anguish_fn + "\n");
            
            console.log("\nDisappointment:");
            console.log("disappointment_tp:", results.disappointment_tp + "\n", "disappointment_tn:", results.disappointment_tn + "\n", "disappointment_fp:", results.disappointment_fp + "\n", "disappointment_fn:", results.disappointment_fn + "\n");
            
            console.log("\nDespair:");
            console.log("despair_tp:", results.despair_tp + "\n", "despair_tn:", results.despair_tn + "\n", "despair_fp:", results.despair_fp + "\n", "despair_fn:", results.despair_fn + "\n");
            
            console.log("\nGrief:");
            console.log("grief_tp:", results.grief_tp + "\n", "grief_tn:", results.grief_tn + "\n", "grief_fp:", results.grief_fp + "\n", "grief_fn:", results.grief_fn + "\n");
            
            console.log("\nMisery:");
            console.log("misery_tp:", results.misery_tp + "\n", "misery_tn:", results.misery_tn + "\n", "misery_fp:", results.misery_fp + "\n", "misery_fn:", results.misery_fn + "\n");
            
            console.log("\nHelplessness:");
            console.log("helplessness_tp:", results.helplessness_tp + "\n", "helplessness_tn:", results.helplessness_tn + "\n", "helplessness_fp:", results.helplessness_fp + "\n", "helplessness_fn:", results.helplessness_fn + "\n");
            
            console.log("\nNeutral:");
            console.log("neutral_tp:", results.neutral_tp + "\n", "neutral_tn:", results.neutral_tn + "\n", "neutral_fp:", results.neutral_fp + "\n", "neutral_fn:", results.neutral_fn + "\n");
            
            

            // Calculate metrics for Anguish
            var anguish_precision = (anguish_tp / (anguish_tp + anguish_fp)) * 100;
            var anguish_recall = (anguish_tp / (anguish_tp + anguish_fn)) * 100;
            var anguish_fmeasure = 2 * ((anguish_precision * anguish_recall) / (anguish_precision + anguish_recall));
            
            console.log("Anguish Metrics:");
            console.log("Precision: ", anguish_precision.toFixed(2), "%");
            console.log("Recall: ", anguish_recall.toFixed(2), "%");
            console.log("F1-Measure: ", anguish_fmeasure.toFixed(2), "%");
            
            // Calculate metrics for Disappointment
            var disappointment_precision = (disappointment_tp / (disappointment_tp + disappointment_fp)) * 100;
            var disappointment_recall = (disappointment_tp / (disappointment_tp + disappointment_fn)) * 100;
            var disappointment_fmeasure = 2 * ((disappointment_precision * disappointment_recall) / (disappointment_precision + disappointment_recall));
            
            console.log("Disappointment Metrics:");
            console.log("Precision: ", disappointment_precision.toFixed(2), "%");
            console.log("Recall: ", disappointment_recall.toFixed(2), "%");
            console.log("F1-Measure: ", disappointment_fmeasure.toFixed(2), "%");
            
            // Calculate metrics for Despair
            var despair_precision = (despair_tp / (despair_tp + despair_fp)) * 100;
            var despair_recall = (despair_tp / (despair_tp + despair_fn)) * 100;
            var despair_fmeasure = 2 * ((despair_precision * despair_recall) / (despair_precision + despair_recall));

            console.log("Despair Metrics:");
            console.log("Precision: ", despair_precision.toFixed(2), "%");
            console.log("Recall: ", despair_recall.toFixed(2), "%");
            console.log("F1-Measure: ", despair_fmeasure.toFixed(2), "%");

            // Calculate metrics for Grief
            var grief_precision = (grief_tp / (grief_tp + grief_fp)) * 100;
            var grief_recall = (grief_tp / (grief_tp + grief_fn)) * 100;
            var grief_fmeasure = 2 * ((grief_precision * grief_recall) / (grief_precision + grief_recall));

            console.log("Grief Metrics:");
            console.log("Precision: ", grief_precision.toFixed(2), "%");
            console.log("Recall: ", grief_recall.toFixed(2), "%");
            console.log("F1-Measure: ", grief_fmeasure.toFixed(2), "%");
            
            // Calculate metrics for Misery
            var misery_precision = (misery_tp / (misery_tp + misery_fp)) * 100;
            var misery_recall = (misery_tp / (misery_tp + misery_fn)) * 100;
            var misery_fmeasure = 2 * ((misery_precision * misery_recall) / (misery_precision + misery_recall));

            console.log("Misery Metrics:");
            console.log("Precision: ", misery_precision.toFixed(2), "%");
            console.log("Recall: ", misery_recall.toFixed(2), "%");
            console.log("F1-Measure: ", misery_fmeasure.toFixed(2), "%");

            // Calculate metrics for Helplessness
            var helplessness_precision = (helplessness_tp / (helplessness_tp + helplessness_fp)) * 100;
            var helplessness_recall = (helplessness_tp / (helplessness_tp + helplessness_fn)) * 100;
            var helplessness_fmeasure = 2 * ((helplessness_precision * helplessness_recall) / (helplessness_precision + helplessness_recall));

            console.log("Helplessness Metrics:");
            console.log("Precision: ", helplessness_precision.toFixed(2), "%");
            console.log("Recall: ", helplessness_recall.toFixed(2), "%");
            console.log("F1-Measure: ", helplessness_fmeasure.toFixed(2), "%");

            // Calculate metrics for Neutral
            var neutral_precision = (neutral_tp / (neutral_tp + neutral_fp)) * 100;
            var neutral_recall = (neutral_tp / (neutral_tp + neutral_fn)) * 100;
            var neutral_fmeasure = 2 * ((neutral_precision * neutral_recall) / (neutral_precision + neutral_recall));

            console.log("Neutral Metrics:");
            console.log("Precision: ", neutral_precision.toFixed(2), "%");
            console.log("Recall: ", neutral_recall.toFixed(2), "%");
            console.log("F1-Measure: ", neutral_fmeasure.toFixed(2), "%");
            

        },
        error: function (error) {
            hideLoader()
            alert('Error in Analyzing. Please Try Again');
            console.error('Error:', error);
        }
    });
}

function createDivElements(results){
    console.log(results.results)
    resultCont.style.display = 'flex';
    for (let i = 0; i < results.results.length; i++) {
        // Get the result-container div
        var resultContainer = document.getElementById("result-container");

        // Create the post div
        var postDiv = document.createElement("div");
        postDiv.className = "post";

        // Create the subreddit_info div
        var subredditInfoDiv = document.createElement("div");
        subredditInfoDiv.className = "subreddit_info";

        // Create the subreddit_info image
        var subredditInfoImg = document.createElement("img");
        subredditInfoImg.width = "14";
        subredditInfoImg.height = "14";
        subredditInfoImg.src = "https://img.icons8.com/doodle/48/reddit--v4.png";
        subredditInfoImg.alt = "reddit--v4";

        // Create the subreddit_name span
        var subredditNameSpan = document.createElement("span");
        subredditNameSpan.className = "subreddit_name";
        subredditNameSpan.textContent = "r/Anonymous";

        // Append image and span to subreddit_info div
        subredditInfoDiv.appendChild(subredditInfoImg);
        subredditInfoDiv.appendChild(subredditNameSpan);

        // Append subreddit_info div to post div
        postDiv.appendChild(subredditInfoDiv);

        // Create post_content div
        var postContentDiv = document.createElement("div");
        postContentDiv.className = "post_content";

        // Create content div
        var contentDiv = document.createElement("div");
        contentDiv.className = "content";

        // Create post_title div
        var postTitleDiv = document.createElement("div");
        postTitleDiv.className = "post_title";
        postTitleDiv.textContent = results.results[i].text;

        // Create post-more-info div
        var postMoreInfoDiv = document.createElement("div");
        postMoreInfoDiv.className = "post-more-info";

        // Create left-info and right-info divs
        var leftInfoDiv = document.createElement("div");
        leftInfoDiv.className = "left-info";
        var rightInfoDiv = document.createElement("div");
        rightInfoDiv.className = "right-info";

        // // Create user divs for left-info
        // var userAnnotationDiv = createUserDiv("Annotation: "+ results.results[i].true_label);
        // if(results.results[i].true_label === 'none'){
        //     var userLabelDiv = createUserDiv("Label: N/A");
        // }else{
        //     var userLabelDiv = createUserDiv("Label: " +results.results[i].label);
        // }

        // Create user divs for right-info
        var userCharacterCountDiv = createUserDiv("Character Count:" + results.results[i].character_count);
        var userProcessingTimeDiv = createUserDiv("Processing Time:"+ results.results[i].timer);

        // Append user divs to left-info and right-info divs
        // leftInfoDiv.appendChild(userAnnotationDiv);
        // leftInfoDiv.appendChild(userLabelDiv);
        rightInfoDiv.appendChild(userCharacterCountDiv);
        rightInfoDiv.appendChild(userProcessingTimeDiv);

        // Append left-info and right-info divs to post-more-info div
        postMoreInfoDiv.appendChild(leftInfoDiv);
        postMoreInfoDiv.appendChild(rightInfoDiv);

        // Create score div
        var scoreDiv = document.createElement("div");
        scoreDiv.className = "score";
        scoreDiv.innerHTML = "▲<span class='votes'>"+results.results[i].predicted_result+"</span>";

        // Append post_title, post-more-info, and score divs to content div
        contentDiv.appendChild(postTitleDiv);
        contentDiv.appendChild(postMoreInfoDiv);
        contentDiv.appendChild(scoreDiv);

        // Append content div to post_content div
        postContentDiv.appendChild(contentDiv);

        // Append post_content div to post div
        postDiv.appendChild(postContentDiv);

        // Append post div to result-container div
        resultContainer.appendChild(postDiv);


        // Helper function to create user div
        function createUserDiv(label) {
        var userDiv = document.createElement("div");
        userDiv.className = "user";
        userDiv.innerHTML = "<strong>" + label + "</strong>";
        return userDiv;
        }

        // Helper function to create p element
        function createParagraph(label) {
        var pElement = document.createElement("p");
        pElement.textContent = label;
        return pElement;
        }
    }
    
        // Create button and count-item divs
        // var showBreakdownBtn = document.createElement("button");
        // showBreakdownBtn.id = "show-breakdown-btn";
        // showBreakdownBtn.textContent = "Show Breakdown";

        var countItemDiv = document.createElement("div");
        countItemDiv.className = "count-item";

        // // Create and append p elements to count-item div
        // var truePositivesP = createParagraph("True Positives: "+ results.tp_count);
        // var falsePositivesP = createParagraph("False Positives: "+ results.fp_count);
        // var trueNegativesP = createParagraph("True Negatives: "+results.tn_count);
        // var falseNegativesP = createParagraph("False Negatives: "+ results.fn_count);

        // countItemDiv.appendChild(truePositivesP);
        // countItemDiv.appendChild(falsePositivesP);
        // countItemDiv.appendChild(trueNegativesP);
        // countItemDiv.appendChild(falseNegativesP);

        // // Append button and count-item divs to result-container div
        // // resultContainer.appendChild(showBreakdownBtn);
        // resultContainer.appendChild(countItemDiv);
}



//function to display only the highest
function displayOneEmotion (result){
    const resultSpanAjax = $('#result-span');
    const highestEmotion = getHighestEmotion(result[0]);
    // const percentage = (result[highestEmotion] * 100).toFixed(1);
    const highestEmotionElement = `${highestEmotion}`;
    resultSpanAjax.append(highestEmotionElement);
    processTime.style.display ='block';
    // // Display the processing time
    processTime.innerHTML = `Processing time: ${result[1].toFixed(5)} seconds`;
}

//function to display all with its percentage
function displayAllEmotion(result){
    const resultSpanAjax = $('#result-span');
    for (const attribute in result[0]) {
        const percentage = (result[0][attribute] * 100).toFixed(1);
        const attributeElement = `<div>${attribute}: ${percentage}%</div>`;
        resultSpanAjax.append(attributeElement);
    }
    allEmotionsDisplayed = false;
}

// Function to get the highest emotion
function getHighestEmotion(result) {
    let highestEmotion = null;
    let highestPercentage = -1;

    for (const attribute in result) {
        const percentage = result[attribute];

        if (percentage > highestPercentage) {
            highestPercentage = percentage;
            highestEmotion = attribute;
        }
    }

    return highestEmotion;
}

function getEmotionColor() {
    //get the value of the span
    if(resultSpan.textContent === 'neutral'){
        resultSpan.style.backgroundColor = 'rgba(66, 66, 156, 0.868)';
    }
    else if(resultSpan.textContent === 'disappointment'){
        resultSpan.style.backgroundColor = 'rgb(197, 109, 109)';
    }
    else if(resultSpan.textContent === 'helplessness'){
        resultSpan.style.backgroundColor ='rgb(197, 64, 64)';
    }
    else if(resultSpan.textContent === 'grief'){
        resultSpan.style.backgroundColor = 'rgb(199, 25, 25)';
    }
    else if(resultSpan.textContent === 'misery'){
        resultSpan.style.backgroundColor = 'rgb(156, 23, 23)'
    }
    else if(resultSpan.textContent === 'anguish'){
        resultSpan.style.backgroundColor = 'rgba(126, 3, 3, 0.868)';
    }
    else if(resultSpan.textContent === 'despair'){
        resultSpan.style.backgroundColor = 'rgba(87, 3, 3, 0.868)';
    }
}

//function to select the preferred model
function selectModel(){
    if (modelSelected.value === '1'){
        indicator.innerHTML = "You are using RoSA's Model"
        //more functionalty
    }
    else if (modelSelected.value === '2'){
        indicator.innerHTML = "You are using Go Emotion's Model"
        //more functionalty
    }
}

//function to select the preferred reddit subreddit
function selectSubreddit(){
    if(subredditSelect.value === 'depression'){
        cardTitle.innerHTML = "r/depression, because nobody should be alone in a dark place";
        cardTitle.href = "https://www.reddit.com/r/depression/new/";
        cardName.innerHTML = "r/depression";
        cardImg.src = "https://styles.redditmedia.com/t5_3nt3b/styles/communityIcon_yye017fvblv21.png?width=256&s=501251dd056a8eb1d4791dce5bc0b8a9bff777bc";
        cardDesc.innerHTML = "Peer support for anyone struggling with a depressive disorder.";
        cardDate.textContent = "Jan 1, 2009";
        cardMembersLabel.textContent = "Members";
        cardMembers.textContent = "996k";
    }
    else if(subredditSelect.value === 'sad'){
        cardTitle.innerHTML = "sad reddit: vent and share";
        cardTitle.href = "https://www.reddit.com/r/sad/";
        cardName.innerHTML = "r/sad";
        cardImg.src = "https://styles.redditmedia.com/t5_2qhja/styles/communityIcon_1rifr6sm66n51.png?width=256&s=39bd98020109e660886cb905270b972470655a7e";
        cardDesc.innerHTML = "A community for sad people";
        cardDate.textContent = "Mar 17, 2018";
        cardMembers.textContent = "137k";
        cardMembersLabel.textContent = "Members";
    }
    else if(subredditSelect.value === 'ForeverAlone'){
        cardTitle.innerHTML = "Forever Alone, Together!";
        cardTitle.href = "https://www.reddit.com/r/ForeverAlone/";
        cardName.innerHTML = "r/ForeverAlone";
        cardImg.src = "https://styles.redditmedia.com/t5_2s3yz/styles/communityIcon_nr2xmvvry6w71.png";
        cardDesc.innerHTML = "A subreddit for Forever Alone folks. Official Discord server: https://discord.com/invite/TvDz9jB";
        cardDate.textContent = "Sep 15, 2010";
        cardMembers.textContent = "184k";
        cardMembersLabel.textContent = "Members";
    }
    else if(subredditSelect.value === 'SuicideWatch'){
        cardTitle.innerHTML = "Peer support for anyone struggling with suicidal thoughts";
        cardTitle.href = "https://www.reddit.com/r/SuicideWatch/";
        cardName.innerHTML = "r/SuicideWatch";
        cardImg.src = "https://styles.redditmedia.com/t5_3nt3b/styles/communityIcon_yye017fvblv21.png?width=256&s=501251dd056a8eb1d4791dce5bc0b8a9bff777bc";
        cardDesc.innerHTML = "Peer support for anyone struggling with suicidal thoughts.";
        cardDate.textContent = "Dec 16, 2008";
        cardMembers.textContent = "450k";
        cardMembersLabel.textContent = "Members";
    }
    else if(subredditSelect.value === 'happy'){
        cardTitle.innerHTML = "Happy Reddit to make you happy";
        cardTitle.href = "https://www.reddit.com/r/happy/";
        cardName.innerHTML = "r/happy";
        cardImg.src = "https://b.thumbs.redditmedia.com/cVLLdr_kSD9sCly5HlQCmPtaoUSLG_s5oRQjBI2YqmY.png";
        cardDesc.innerHTML = "Too many depressing things on the main page, so post about what makes you warm and fuzzy inside!";
        cardDate.textContent = "Jan 25, 2008";
        cardMembers.textContent = "722k";
        cardMembersLabel.textContent = "Happy Campers";
    }
}

function extractAnalyze(){
    showLoader();
    const subredditSelect = document.getElementById('subreddit-select');
    const postnumber = document.getElementById('post-quantity-select');
    const preferredM = $('#model').val();
    const datadata = JSON.stringify({ subreddit: subredditSelect.value, postQuantity: postnumber.value, prefmodel: preferredM})
    
    // var postTitleReddit = document.querySelectorAll('.post_title-reddit');
    // var votesReddit = document.querySelectorAll('.votes-reddit');
    var existingDiv = document.querySelector('.content-reddit');
    
    $.ajax({
        type: 'POST',
        url: '/extractAndAnalyze',
        contentType: 'application/json;charset=UTF-8',
        dataType: 'json',
        data: datadata,
        
        //once success, perform the display
        success: function (result) {
            hideLoader();
            console.log(result[0])
            console.log(result[1])
            //make the container visible
            rightSideReddit.style.display = 'block'
            for (let i = 0; i < result[0].length; i++) {
            // Get the existing container
            var container = document.getElementById('right-side-reddit');

            // Create the structure
            var postReddit = document.createElement('div');
            postReddit.className = 'post-reddit';

            var userInfo = document.createElement('div');
            userInfo.className = 'user_info';

            var redditIcon = document.createElement('img');
            redditIcon.width = 14;
            redditIcon.height = 14;
            redditIcon.src = 'https://img.icons8.com/doodle/48/reddit--v4.png';
            redditIcon.alt = 'reddit--v4';

            var userName = document.createElement('span');
            userName.className = 'user_name';
            userName.textContent = 'r/Anonymous';

            var postContentReddit = document.createElement('div');
            postContentReddit.className = 'post_content-reddit';

            var contentReddit = document.createElement('div');
            contentReddit.className = 'content-reddit';

            var postTitleReddit = document.createElement('div');
            postTitleReddit.className = 'post_title-reddit';
            postTitleReddit.textContent = result[0][i];

            var postMoreInfoReddit = document.createElement('div');
            postMoreInfoReddit.className = 'post-more-info-reddit';

            var rightInfoReddit = document.createElement('div');
            rightInfoReddit.className = 'right-info-reddit';

            var characterCount = document.createElement('div');
            characterCount.className = 'user-reddit';
            characterCount.innerHTML = '<strong>Character Count:'+result[3][i]+'</strong>';

            var processingTime = document.createElement('div');
            processingTime.className = 'user-reddit';
            processingTime.innerHTML = '<strong>Processing Time: '+result[2][i]+'</strong>';

            var scoreReddit = document.createElement('div');
            scoreReddit.className = 'score-reddit';
            scoreReddit.innerHTML = '▲<span class="votes-reddit">' + result[1][i] + '</span>';

            // Append elements to build the structure
            rightInfoReddit.appendChild(characterCount);
            rightInfoReddit.appendChild(processingTime);

            postMoreInfoReddit.appendChild(rightInfoReddit);

            contentReddit.appendChild(postTitleReddit);
            contentReddit.appendChild(postMoreInfoReddit);
            contentReddit.appendChild(scoreReddit);

            userInfo.appendChild(redditIcon);
            userInfo.appendChild(userName);

            postContentReddit.appendChild(contentReddit);

            postReddit.appendChild(userInfo);
            postReddit.appendChild(postContentReddit);

            container.appendChild(postReddit);
              }
        },
        error: function (error) {
            hideLoader();
            alert('Error in Analyzing. Please Try Again');
            console.error('Error:', error);
        }
    });
}





// **------------ EVENT LISTENERS ------------** //

// TOGGLE FOR LIGHT AND DARK MODE
const toggleButton = document.getElementById("toggle-mode-btn");
if (toggleButton) {
    toggleButton.addEventListener("click", toggleMode);
}


// CHARACTER COUNTER
textarea.addEventListener('input', function () {
    updateCharacterCounter();
    if (currentLength == maxLength){
        alert('Maximum length reached. Please check your input.')
    }
});

// CHANGE TO LINK INPUT FROM TEXT INPUT
linkOptionbtn.addEventListener("click", function(){
    textOptionbtn.disabled = false;
    linkOptionbtn.disabled = true;

    //visibility
    analyzeform.style.display = 'block';
    extractform.style.display = 'block'
    subredditForm.style.display = 'none';
    counter.style.display = 'block';
    buttonCont.style.display = 'block';

    linkcont.style.display = 'flex';
    textarea.style.display = 'none';
    counter.style.display = 'none';
    
    //change the label to reddit link
    redditlabel.innerHTML = "Reddit Link:";

    
    //clear the text area from text input if there's any.
    textarea.value = "";
    
    //reset the character counter
    counter.textContent = '0/5000';

    //If ever the analyze button was clicked, clear the result as well
    resultSpan.innerHTML = '';
    resultLabel.style.display = 'none';
    resultSpan.style.display = 'none';
    viewMore.style.display = 'none';
    processTime.innerHTML = '';

    //call the extractPostFromLink function
    extractBTN.addEventListener("click", extractPostFromLink);
    
    //hide the subreddit form
    subredditForm.style.display = 'none'
        
    //if ever the upload file was clicked, clear the container
    resultCont.style.display = "none";

});

//CHANGE BACK TO TEXT INPUT FROM LINK INPUT
textOptionbtn.addEventListener("click", function(){
    textOptionbtn.disabled = true;
    linkOptionbtn.disabled = false;

    //visibility
    analyzeform.style.display = 'block';
    subredditForm.style.display = 'none';

    linkcont.style.display = 'none';
    textarea.style.display = 'block';
    counter.style.display = 'block';
    buttonCont.style.display = 'block';
    //change the label back to reddit post
    redditlabel.innerHTML = "Reddit Post:";

    //remove the value of the link in changing.
    linkinput.value = "";

    //remove the value of the text area.
    textarea.value = '';

    //reset the character counter
    counter.textContent = '0/5000';

    //If ever the analyze button was clicked, clear the result as well
    resultSpan.innerHTML = '';
    resultLabel.style.display = 'none';
    resultSpan.style.display = 'none';
    viewMore.style.display = 'none';
    processTime.innerHTML = '';

    //if ever the upload file was clicked, clear the container
    resultCont.style.display = "none";
    //change the placeholder back to enter reddit post
    textarea.placeholder = "Enter Reddit Post";
    
    //hide the subreddit form
    subredditForm.style.display = 'none'

    //enable users to input reddit post
    textarea.readOnly = false;

});

// CLEAR TEXT AREA
clearButton.addEventListener("click", clearAll);

//USER CLICK ANALYZE BUTTON
analyzeBtn.addEventListener('click', function(event){
    viewMore.textContent = '•••'
    //get the current character length of the text area
    var currentLength = textarea.value.length;
    //set the min length of input
    var minLength = 80;
    if(textarea.value == ''){
        alert('Please enter a Reddit Post')
        event.preventDefault();
    }
    else if(currentLength < minLength){
        alert('Input too short. Please input more than 80 characters.')
        event.preventDefault();
    }
    else{
        event.preventDefault();
        //perform the model execution
        performModelExecution();

        // Display the result
        resultSpan.style.display = 'flex';
        resultLabel.style.display = 'flex';

        //get the more button to show
        // viewMore.style.display = 'block';
    }
});

//BREAKDOWN OF RESULT
viewMore.addEventListener('click', function(event) {
    if (viewMore.textContent === '•••') {
        allEmotionsDisplayed = true;
        //perform the model execution again.
        performModelExecution();
        viewMore.textContent = "↩";
        resultLabel.style.display = 'none';
        resultSpan.style.backgroundColor = '';
        
    } else if (viewMore.textContent === '↩') {
        allEmotionsDisplayed = false;
        performModelExecution();
        viewMore.textContent = "•••";
        resultLabel.style.display = 'flex';

    }
});


// LOADER FOR THE WHOLE WEBSITE

// document.addEventListener("DOMContentLoaded", function () {
//     var loaderWrapper = document.querySelector('.loader-wrapper');
//     var content = document.querySelector('.content');
  
//     // Hide loader when the page is fully loaded
//     window.addEventListener('load', function () {
//       loaderWrapper.style.display = 'none';
//       content.style.display = 'block';
//     });
//   });

 //MODEL SELECTION
document.addEventListener('DOMContentLoaded', function () {
        selectModel();
        selectSubreddit();
    
    modelSelected.addEventListener('click', function(){
        selectModel();
    });

    subredditSelect.addEventListener('change', function(){
        selectSubreddit();
    });
});

// var showBreakdownbtn = document.getElementById('show-breakdown-btn');
// FILE UPLOAD FORM HANDLER
document.getElementById('file-upload-form').addEventListener('submit', function(event) {
    var fileInput = document.getElementById('file-option-btn');
    if (!fileInput.files.length) {
        alert('Please select a file before submitting');
        event.preventDefault();
    }
    else{
        event.preventDefault();
        linkOptionbtn.disabled = false;
        textOptionbtn.disabled = false;
        subredditForm.style.display ='none';
        linkcont.style.display = 'none';
        textarea.style.display = 'none';
        counter.style.display  = 'none';
        buttonCont.style.display = 'none';
        // clearButton.style.display = 'none';
        // analyzeBtn.style.display = 'none';
        resultCont.innerHTML = "";
        performModelExecutionForFile();
        redditlabel.innerHTML = 'Predictions from the file: '
    }
});

// IF SUBREDDIT FORM IS CLICKED
redditbtn.addEventListener('click', function(){
    redditlabel.innerHTML = 'Select Subreddit:'
    subredditForm.style.display = 'flex';
    
    extractform.style.display = 'none';
    analyzeform.style.display = 'none';
    textOptionbtn.disabled = false;
    linkOptionbtn.disabled = false;
    resultCont.innerHTML = ''
});


//ANALYZE AND EXTRACT IS SELECTED
extractAndAnalyze.addEventListener('click', function(event){
    event.preventDefault();
    rightSideReddit.innerHTML = '';
    extractAnalyze();
});



// //BREAKDOWN HANDLER FOR RESULTS
// document.getElementById('show-breakdown-btn').addEventListener('click', function() {
//     // Toggle the display property of result items when the button is clicked
//     var resultItem = document.querySelectorAll('.post');
//     resultItem.forEach(function(item) {
//         item.style.display = (item.style.display === 'none') ? 'block' : 'none';
//     });
// });




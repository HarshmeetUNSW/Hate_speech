import streamlit as st
from PIL import Image
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

# Load an image
logo = Image.open("TikTok-logo.png")

# Set up the layout
col1, col2 = st.columns([1, 3])
with col1:
    st.image(logo, width=100)  # Adjust width as needed

with col2:
    st.title('Hate Speech Detector')

# Load the model and tokenizer once
@st.cache_resource 
def load_model():
    model = BertForSequenceClassification.from_pretrained('final_fine_tuned_bert/')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.eval()  # Put the model in evaluation mode
    return model, tokenizer

model, tokenizer = load_model()

# Initialize LIME Text Explainer
explainer = LimeTextExplainer(class_names=["Hateful", "Normal", "Offensive"])

def predict_and_explain(text):
    # Prepare the text input for BERT
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=180)

    # Disable gradient calculation
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predictions
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    max_prob, predicted_label = torch.max(probabilities, dim=1)

    # Generate LIME explanation
    exp = explainer.explain_instance(text, 
                                     lambda x: torch.softmax(model(**tokenizer(x, return_tensors='pt', padding=True, truncation=True)).logits, dim=1).detach().numpy(),
                                     num_features=6,
                                     num_samples=100,  # Adjust the number of samples for faster execution if needed
                                     top_labels=1)

    # Display LIME explanation in the Streamlit app
    # st.markdown("### Text with highlighted words")
    fig = exp.as_pyplot_figure(label=predicted_label.item())
    # st.pyplot(fig)
    #st.markdown(exp.as_html(), unsafe_allow_html=True)


    return predicted_label, max_prob.numpy()[0], fig

def on_predict():
    # Handler for the predict button
    label, confidence, fig = predict_and_explain(st.session_state.text)
    emoji, label_description = get_emoji(label)
    confidence = round(confidence * 100, 2)
    st.session_state.results = f'{emoji}  {label_description} with Confidence: **{confidence}%**'
    st.markdown("### Text with highlighted words")

    st.pyplot(fig)



def get_emoji(label):
    if label == 0:
        return "😡", "Hateful"  # Update label descriptions as per your labels
    elif label == 1:
        return "😊", "Normal"
    else:
        return "😠", "Offensive"

# Text input
st.text_area("Enter Text for Prediction", key="text", on_change=on_predict, args=())

# Predict button
st.button("Predict", on_click=on_predict)

# Display results
if "results" in st.session_state:
    st.markdown(st.session_state.results, unsafe_allow_html=True)

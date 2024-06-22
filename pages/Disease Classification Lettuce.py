import streamlit as st
from PIL import Image
from utils.utils import extract_json
from utils.model import classify_image,recomendations
import os
 
#page config only change the plant_name in the page_title 
st.set_page_config(
    page_title="Lettuce Dissease Classificaition",
    layout="wide", 
)
#CHANGE THIS ACCORDINGLY TO THE PLANT NAME
plant_name = "Lettuce"

#INIT VARIABLES
#CHANGE json_path (MODEL METADATA) to extract model metadata in sidebar
json_path = './resources/models/lettuce_modelmetadata.json'
model_metadata = extract_json(json_path)
energy = model_metadata['energy']
#change general recomendation for disease
general_recommendations = "general recommendations: lorem ipsum"
#change class id to get name (match the prediction result to number)
class_idx_to_name_dict  = {
    0: "Bacterial Infection",
    1: "Fungal Infection",
    2: "Healthy Lettuce"
}
#change sample image path
sample_image_path = "./resources/sample_images/lettuce"
#change the xlsx of recomendation (always use the template format, pass the rows as 'No Data' if here isn't any justified action for the disease)
rec_path = './resources/actionable/lettuce recommendations.xlsx'
#change model path
model_path = "resources/models/model_lettuce.pth"


#side bar
st.sidebar.markdown(f"{plant_name} Dissease Classificaition")
st.sidebar.write(f":red-background[MODEL NAME     : {model_metadata['model_name']}]")
st.sidebar.write(f":red-background[Train F1-Macro : {model_metadata['Train F1-macro']}]")
st.sidebar.write(f":red-background[Test F1-Macro  : {model_metadata['Test F1-macro']}]")

#title bar
st.title("OMDENA MILAN AGRITECH: TASK 3")
st.markdown(f"# {plant_name} Dissease Classification")
st.subheader(':orange[Choose other classification task via side bar]')
st.divider()

#body

#data
example_images = os.listdir(sample_image_path)
example_images = [os.path.join(sample_image_path, img) for img in example_images]

#body widgets
st.header('You can classify by choose an example or upload a photo')
st.subheader("Don't choose both")


#first row
sample_data = st.selectbox("Choose an example image:", ['None'] + example_images)

#second row
uploaded_data = st.file_uploader(f"Choose an image of {plant_name})", type="jpg")

button = st.button('Classify!!!')
if button:
    if sample_data != 'None' and uploaded_data is None:
        image_data = sample_data
    elif uploaded_data is not None and sample_data == 'None':
        image_data = uploaded_data
    else:
        st.title(':red[PLEASE, UPLOAD YOUR IMAGE OR CHOOSE A SAMPLE IMAGE]')
    try:   
        image = Image.open(image_data)
        st.image(image, caption="Selected Example Image", use_column_width=True)
        result = classify_image(image, energy, model_path)
        recomendations(result, general_recommendations, class_idx_to_name_dict, rec_path,energy)
    except NameError:
        st.title(':red[PLEASE, UPLOAD YOUR IMAGE OR CHOOSE A SAMPLE IMAGE]')
    image_data = 'None'
    uploaded_data = None
    sample_data = None




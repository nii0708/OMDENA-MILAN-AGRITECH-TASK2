import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import pandas as pd
import torch

def load_recommendations(path):
    recommendations = pd.read_excel(path)
    return recommendations

def transform_image(image, device):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image = transform(image=np.array(image))['image']
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)
    return image_tensor

def calculate_probability_and_predicted_class(output):
    p = torch.softmax(output, dim=1)
    probability, predicted_class = torch.max(p, 1)
    return probability, predicted_class

def calculate_energy(output):
    return -torch.logsumexp(output, dim=1)

def load_model(path, device):
    model = torch.load(path, map_location=device)
    model.eval()
    return model

def classify_image(image, 
                   energy,
                   model_path = "resources/models/model_lettuce.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    model.to(device)
    result = inference(model, image, device, energy)
    return result

def recomendations(result,
                   general_recommendations,
                   class_idx_to_name_dict,
                   rec_path,
                   energy):
    recommendations = load_recommendations(rec_path)

    if result['result'] == 'unknown':
        st.write("The class is of unknown origin")
        st.write(f"The energy is {result['energy']:.4f}, but the energy threshold is {energy:.4f}")
        st.write(f"Potentially it could be: {class_idx_to_name_dict[result['predicted_class']]} with the probability {result['probability']:.4f}")
        st.write("General Recommendations for Cucumber:")
        st.write(general_recommendations)
    else:
        st.write(f":blue-background[Probability: {result['probability']:.4f}]")
        st.write(f":blue-background[Predicted class: {class_idx_to_name_dict[result['predicted_class']]}]")
        if class_idx_to_name_dict[result['predicted_class']] not in ["Healthy_Crop_Cucumber", "Healthy_Crop_Leaf"]:
            st.write("Recommendations")
            st.write("Pesticide Methods:")
            st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['pesticide'].values[0])
            st.write("Non-pesticide Methods:")
            st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['non-pesticide'].values[0])
        else:
            st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['maintenance'].values[0])

def inference(model, image, device, energy):

    image_tensor = transform_image(image, device)

    with torch.inference_mode():
        output = model(image_tensor)
        energy = calculate_energy(output)
        energy_value = energy.item()
        probability, predicted_class = calculate_probability_and_predicted_class(output)
        probability_value = probability.item()
        predicted_class_value = predicted_class.item()
        #return energy_value, probability_value, predicted_class_value
        if energy_value > energy:            
            
            return {
                'result': 'unknown',
                'energy': energy_value,
                'probability': probability_value,
                'predicted_class': predicted_class_value
            }

        return {
            'result': 'known',
            'probability': probability_value,
            'predicted_class': predicted_class_value
        }
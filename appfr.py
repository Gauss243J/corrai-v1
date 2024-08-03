import streamlit as st
import openai
import base64
import pandas as pd
from io import BytesIO
from PIL import Image
import io
import re


hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Function to convert image to base64
def image_to_base64(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Function to display an image from base64
def display_image_from_base64(encoded_image):
    image_data = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_data))
    return image

# Function to extract content from an image using GPT-4 Vision
def extract_content_from_image(encoded_image, api_key):
    openai.api_key = api_key
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extraire le contenu de cette image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    content = response['choices'][0]['message']['content']
    return content

# Function to grade a student copy based on the reference
def grade_student_copy(reference_content, student_content, api_key):
    openai.api_key = api_key
    
    prompt = f"""
    Réponse de référence :
    {reference_content}

    Réponse de l'étudiant :
    {student_content}

    Veuillez effectuer les tâches suivantes :
    1. Identifier le nom de l'étudiant à partir de sa réponse.
    2. Évaluer la réponse de l'étudiant en fonction de sa précision et de son exactitude par rapport à la réponse de référence. Fournir une note sur 100 et un court commentaire.

    Formatez votre réponse comme suit :
    Nom : [nom de l'étudiant]
    Note : [0-100]
    Commentaire : [court commentaire]
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that grades student answers based on a reference answer, and what identifies student names."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.1,
        top_p=1
    )

    try:
        content = response['choices'][0]['message']['content'].strip()
        name_line, score_line, feedback_line = content.split('\n')

        name = name_line.split(":", 1)[1].strip()
        score = int(''.join(filter(str.isdigit, score_line)))
        feedback = feedback_line.split(":", 1)[1].strip()

    except (KeyError, IndexError, ValueError):
        name = "Inconnu"
        score = "Erreur"
        feedback = "Erreur dans la génération de la note ou la détection du nom."

    return name, score, feedback

# Function to generate CSV from the results
def to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue().decode('utf-8')

# Main Streamlit application function
st.title("CorrAI : Système de correction des Copies")

# Example extracted content with multiple LaTeX formulas
extracted_content = r'''
Voici la première formule LaTeX :
\[
    \frac{\partial f(x,t)}{\partial x} + \frac{\partial f(x,t)}{\partial t} = 0
\]
et du texte après cela. Voici une autre formule :
\[
    E = mc^2
\]
et encore du texte descriptif. Enfin, une autre formule :
\[
    a^2 + b^2 = c^2
\]
et du texte de conclusion.
'''

def extract_latex_and_text(content):
    # Regex pattern to extract multiple LaTeX blocks and surrounding text
    pattern = r'(?s)(.*?)\\\[([\s\S]*?)\\\]'

    # Find all matches for LaTeX blocks and surrounding text
    matches = re.finditer(pattern, content)

    parts = []
    last_pos = 0

    for match in matches:
        before_latex = match.group(1)
        latex_content = match.group(2)
        
        # Text before LaTeX
        if before_latex.strip():
            parts.append(('text', before_latex.strip()))

        # LaTeX content
        parts.append(('latex', latex_content.strip()))
        last_pos = match.end()

    # Remaining text after the last LaTeX block
    remaining_text = content[last_pos:].strip()
    if remaining_text:
        parts.append(('text', remaining_text))

    return parts

st.header("Télécharger la Copie de Référence")
reference_file = st.file_uploader("Téléchargez l'image de la copie de référence", type=["jpg", "jpeg", "png"])

st.header("Télécharger les Copies des Étudiants")
student_files = st.file_uploader("Téléchargez les images des copies des étudiants", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

api_key = st.secrets["API_KEY"]

if reference_file and student_files and api_key:
    # Convert reference image to base64 and extract content
    reference_image = image_to_base64(reference_file)
    reference_content = extract_content_from_image(reference_image, api_key)
    
    # Display reference image
    st.subheader("Copie de Référence")
    reference_display_image = display_image_from_base64(reference_image)
    st.image(reference_display_image, caption="Copie de Référence")
    st.write("Contenu Extrait de la Copie de Référence :")
    #st.write(reference_content)
    
    # Extract and display content
    parts = extract_latex_and_text(reference_content)

    for part_type, content in parts:
        if part_type == 'text':
            st.write(content)  # Display normal text
        elif part_type == 'latex':
            st.latex(content)  # Display LaTeX content
        
    # Prepare to store the results
    results = []

    # Process each student copy
    for student_file in student_files:
        student_image = image_to_base64(student_file)
        student_content = extract_content_from_image(student_image, api_key)
        
        # Display student image
        st.subheader(f"Copie d'Étudiant : {student_file.name}")
        student_display_image = display_image_from_base64(student_image)
        st.image(student_display_image, caption=f"Copie d'Étudiant : {student_file.name}")
        st.write("Contenu Extrait de la Copie d'Étudiant :")
        #st.write(student_content)
        # Format the LaTeX content
        # Extract and display content
        parts = extract_latex_and_text(student_content)

        for part_type, content in parts:
            if part_type == 'text':
                st.write(content)  # Display normal text
            elif part_type == 'latex':
                st.latex(content)  # Display LaTeX content
                        
        name, score, feedback = grade_student_copy(reference_content, student_content, api_key)
        results.append({"Nom": name, "Note": score, "Commentaire": feedback})

    # Display the results
    st.header("Résultats de l'Évaluation")
    results_df = pd.DataFrame(results)
    st.write(results_df)

    # Provide an option to download the results
    csv_data = to_csv(results_df)
    st.download_button(label="Télécharger les Résultats en CSV", data=csv_data, file_name="resultats_evaluation.csv", mime="text/csv", key="download_csv")

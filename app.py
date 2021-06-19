import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r'C:\Users\ADMIN\Downloads\model_4_fast_text.h5')
    # model.make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    return model

@st.cache
def load_token():
    with open(r'C:\Users\ADMIN\Downloads\tokenizer.pickle', 'rb') as handle:
        tok = pickle.load(handle)
    return tok


if __name__ == '__main__':
    st.markdown(
        '<h1 style="text-align:center;color:white;font-weight:bolder;font-size:100px;background: -webkit-linear-gradient(#e20b0b,#ec720e,#46a3e0,#093ff0); -webkit-background-clip: text;-webkit-text-fill-color: transparent;">Classify Sexual Harrasment Stories</h1>',
        unsafe_allow_html=True)
    st.write("With the recent rise of #MeToo, an increasing number of personal stories about sexual harassment and sexual abuse have been shared online. This case study tries to automatically categorize and analyze various forms of sexual harassment, based on stories shared on online forum SafeCity for the labels of groping/touching, ogling/staring, and commmenting.")
    st.markdown('<h1 style="text-align:left;color:white;font-weight:bolder;font-size:20px;">Describe the incident here :</h1>', unsafe_allow_html=True)
    sentence = st.text_input(' ')
    tokenizer = load_token()
    model = load_model()
    description_sequences = tokenizer.texts_to_sequences([sentence])
    description_padding = pad_sequences(description_sequences, maxlen=300, dtype='int32', padding='post', truncating='post')

    if sentence:
        y_hat = model.predict(description_padding)

        a = ['Commenting', 'Staring', 'Touching']
        b = [i for i in y_hat[0]]

        harrasment = [i[0] for i in list(zip(a, b)) if i[1] >= 0.5]
        s = ''
        for i in harrasment:
            s += i+', '

        if len(harrasment) != 0:
            st.write('Type of Sexual Harrasment : ', s)
        else:
            st.text('No Harrasment')

st.header("Contacts")

"""
[![LinkedIN Connect](https://img.shields.io/badge/Linkedin-connect-blue)](https://www.linkedin.com/in/bhavesh-naidu-23bb52199?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B4EZqZ2Q%2FQd2QiO%2BZEhU68A%3D%3D)
[![GitHub ](https://img.shields.io/badge/Github-source%20code-lightgrey)](https://github.com/bhaveshnaidu999)
[![MAIL](https://img.shields.io/badge/-bhaveshnaidu999@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:bhaveshnaidu999@gmail.com)](mailto:bhaveshnaidu999@gmail.com)
If you happen to have any query or Feedback please connect with me.
"""
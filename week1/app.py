import streamlit as st
import joblib
import pandas as pd
@st.cache(allow_output_mutation=True)
def load(scaler_path, model_path):
    sc = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return sc , model

def inference(row, scaler, model, feat_cols):
    df = pd.DataFrame([row], columns = feat_cols)
    X = scaler.transform(df)
    features = pd.DataFrame(X, columns = feat_cols)
    if (model.predict(features)==0):
        return "the wine is quailty "
    else: return "not quality wine"

st.title('Wine quality Prediction App')
st.write('The data for the following example is for wine quailty prediction.')

#st.image(image, use_column_width=True)
st.write('Please fill in the details of the wine composition')

row = [7.4,0.7, 0, 1.9, 0.076, 11, 34, 0.099,3.51,0.56,9.4]

if (st.button('predict Quality')):
    feat_cols = ['fixed acidity','volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH','sulphates','alcohol']

    sc, model = load('model/scaler.joblib', 'model/model.joblib')
    result = inference(row, sc, model, feat_cols)
    st.write(result)

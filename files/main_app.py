import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import app


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["||___Home Page___||", "||___Test Report and Update___||", "||___Indiviual Patient Update___||",])
    tab1.write("Home Page")
    tab2.write("Test Report and Update")
    tab3.write("Indiviual Patient Update")

    # Tab1- Home Page
    with tab1:
        st.title("Report generator for Parkinsons Patients")
        st.write("\n")
        st.header("Welcome to the information desk. ")
        st.write("\n")
        st.write("\n")
        st.write('Instruction : ')
        st.write("\n")
        st.write("1. Visit Home page for instructions.")
        st.write("2. Visit 'Test Report and Update' page for generating report.")
        st.write("3. Visit 'Indiviual Patient Update' page for collecting individual report.")

    # Tab2- Test Report and Update
    with tab2:

        st.write("Kindly, upload all the complete patient report here.")

        df_file = st.file_uploader('Upload a CSV', type=['csv'], help='Only csv file is acceptable')

        if df_file is not None:
            df = pd.read_csv(df_file)
            st.dataframe(df)
            patient_prediction , full_csv = app.run_app(df)
            csv = convert_df(full_csv)
            st.download_button('Download updated file as CSV', data=csv, file_name='updated_file.csv', mime='text/csv')

        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("Pateint report should have : patient ID, MDVP:Fo (Hz), MDVP:Fhi (Hz), MDVP:Flo (Hz), MDVP:Jitter (%), MDVP:Jitter (Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer (dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2 and PPE.")
        
    # Tab3- Indiviual Patient Update
    with tab3:

        st.write("If you want any individual patient's details kindly input his/her id below. ")

        with st.form(key='patient_form', clear_on_submit = True):
            patient_id = st.text_input('Patient ID') 
            patient_name = st.text_input('Patient Name')
            submitted_button = st.form_submit_button('Submit')

        if submitted_button:
            if patient_id != "":
                    st.subheader("Datails : ")
                    st.write("Pateient ID : ", patient_id)

                    if patient_name != "":
                        st.write("Pateient Name : ", patient_name)

                    p_pred_value = patient_prediction.loc[patient_prediction['Patient ID'] == patient_id, 'Prediction'].item()

                    if patient_id in patient_prediction['Patient ID'].values :
                        st.write("Pateient Report : ", p_pred_value)
                    else:
                        st.write("Pateient Report : Not Available")
                              
      
            else:
                st.write("At least Patient ID is Required")

@st.cache
def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


if __name__ == "__main__":
    main()
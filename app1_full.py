# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

sex_d = {0:"Women",1:"Man"}
pclass_d = {0:"First",1:"Second", 2:"Third"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="Titanic App")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG")

	with overview:
		st.title("Titanic App")

	with left:
		sex_radio = st.radio( "Gender", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		pclass_radio = st.radio( "Class", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
		embarked_radio = st.radio( "Emarked", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )

	with right:
		age_slider = st.slider("Age", value=1, min_value=1, max_value=80)
		sibsp_slider = st.slider("Siblings", min_value=0, max_value=10)
		parch_slider = st.slider("Childrens", min_value=0, max_value=10)
		fare_slider = st.slider("Fare", min_value=0, max_value=500, step=1)

	data = [[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Would I survive?")
		st.subheader(("Yes" if survival[0] == 1 else "No"))
		st.write("Prob {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()

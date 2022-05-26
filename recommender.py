"""
source_path = os.path.join("data/coursera-courses.csv")

User defined functions-

main()
load_data() - called inside main()
preparation_for_cbr(dataframe) - called inside main()

content_based_recommendations(dataframe, inputed_course, courses) - 
called inside preparation_for_cbr(dataframe)
filter(dataframe, chosen_options, column_name, id) - 
called inside preparation_for_cbr(dataframe)

recommendations(dataframe, inputed_course, cosine_similarity_matrix, find_similar=True, how_many=5) - 
called inside content_based_recommendations(dataframe, inputed_course, courses)
extract_keywords(dataframe, column_name) - 
called inside content_based_recommendations(dataframe, inputed_course, courses)

clean_column_names(dataframe, columns) - called inside prepare_data(dataframe)
"""

import streamlit as st
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt 
import altair as alt
import seaborn as sns

from rake_nltk import Rake
from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

#FILTERED_COURSES = None
#SELECTED_COURSE = None

@st.cache(persist=True)
def clean_column_names(dataframe, columns):
	"""
	Function to clean column names
	-----
	columns:
		variable that contains List of column names
	"""
	new = []
	for c in columns:
		new.append(c.lower().replace(' ','_'))
	return new

@st.cache(persist=True)
def prepare_data(dataframe):
	"""
	Function to Prepares the final dataset
	"""
	# clean column names
	dataframe.columns = clean_column_names(dataframe, dataframe.columns)

	# impute missing values that creeped in
	dataframe['skills'] = dataframe['skills'].fillna('Missing')
	dataframe['instructors'] = dataframe['instructors'].fillna('Missing')

	# making certain features numeric
	def make_numeric(x):
		if(x=='Missing'):
			return np.nan
		return float(x)

	dataframe['course_rating'] = dataframe['course_rating'].apply(make_numeric)
	dataframe['course_rated_by'] = dataframe['course_rated_by'].apply(make_numeric)
	dataframe['percentage_of_new_career_starts'] = dataframe['percentage_of_new_career_starts'].apply(make_numeric)
	dataframe['percentage_of_pay_increase_or_promotion'] = dataframe['percentage_of_pay_increase_or_promotion'].apply(make_numeric)

	def make_count_numeric(x):
	    if('k' in x):
	        return (float(x.replace('k','')) * 1000)
	    elif('m' in x):
	        return (float(x.replace('m','')) * 1000000)
	    elif('Missing' in x):
	        return (np.nan)

	dataframe['enrolled_student_count'] = dataframe['enrolled_student_count'].apply(make_count_numeric)

    # extract time to complete
	def find_time(x):
	    l = x.split(' ')
	    idx = 0
	    for i in range(len(l)):
	        if(l[i].isdigit()):
	            idx = i 
	    try:
	        return (l[idx] + ' ' + l[idx+1])
	    except:
	        return l[idx]

	dataframe['estimated_time_to_complete'] = dataframe['estimated_time_to_complete'].apply(find_time)

	# split by skills
	def split_it(x):
		return (x.split(','))
	dataframe['skills'] = dataframe['skills'].apply(split_it)

	return dataframe

@st.cache(allow_output_mutation=True)
def load_data():
	dataset_source_path1 = os.path.join("data/coursera-courses-overview.csv")
	dataset_source_path2 = os.path.join("data/coursera-individual-courses.csv")
	df_overview = pd.read_csv(dataset_source_path1)
	df_individual = pd.read_csv(dataset_source_path2)
	dataframe = pd.concat([df_overview, df_individual], axis=1)

	# preprocess it now
	dataframe = prepare_data(dataframe)

	return dataframe

@st.cache(persist=True)
def filter(dataframe, chosen_options, column_name, id):
	code_selected_url_records = []
	for i in range(1000):
		for op in chosen_options:
			if op in dataframe[column_name][i]:
				code_selected_url_records.append(dataframe[id][i])
	return code_selected_url_records

# def extract_keywords(dataframe, column_name):
#     r = Rake()
#     keyword_lists = []
#     for i in range(1000):
#         descr = dataframe[column_name][i]
#         r.extract_keywords_from_text(descr)
#         key_words_dict_scores = r.get_word_degrees()
#         keywords_string = " ".join(list(key_words_dict_scores.keys()))
#         keyword_lists.append(keywords_string)
        
#     return keyword_lists

def extract_keywords(dataframe, column_name):

    r = Rake()
    keyword_lists = []

    #for i in range(1000):
    for i in range(dataframe[column_name].shape[0]):
        descr = dataframe[column_name][i]
        r.extract_keywords_from_text(descr)
        key_words_dict_scores = r.get_word_degrees()
        keywords_string = " ".join(list(key_words_dict_scores.keys()))
        keyword_lists.append(keywords_string)
        
    return keyword_lists

def recommendations(dataframe, inputed_course, cosine_similarity_matrix, find_similar=True, how_many=5):
    
    # initialise recommended courses list
    recommended = []
    selected_course = dataframe[dataframe['course_name']==inputed_course]
    
    # index of the course fed as input
    s=selected_course.head()
    st.write(s.index)
    idx = selected_course.index[0]

    # creating a Series with the similarity scores in descending order
    if(find_similar):
        score_series = pd.Series(cosine_similarity_matrix[idx]).sort_values(ascending = False)
    else:
        score_series = pd.Series(cosine_similarity_matrix[idx]).sort_values(ascending = True)

    # getting the indexes of the top 'how_many' courses
    if(len(score_series) < how_many):
    	how_many = len(score_series)
    top_suggetions = list(score_series.iloc[1:how_many+1].index)
    
    # populating the list with the titles of the best 10 matching courses
    for i in top_suggetions:
        qualified = dataframe['course_name'].iloc[i]
        recommended.append(qualified)
        
    return recommended

def content_based_recommendations(dataframe, inputed_course, courses):

	# filter out the courses
	dataframe = dataframe[dataframe['course_name'].isin(courses)].reset_index()

	# creating description keywords
	dataframe['descr_keywords'] = extract_keywords(dataframe, 'description')

	# instantiating and generating the count matrix
	count = CountVectorizer()
	count_matrix = count.fit_transform(dataframe['descr_keywords'])

	# generating the cosine similarity matrix
	cosine_similarity_matrix = cosine_similarity(count_matrix, count_matrix)

	# making the recommendation
	recommend_similar_courses = recommendations(dataframe, inputed_course, cosine_similarity_matrix, True)
	temporary_similar = dataframe[dataframe['course_name'].isin(recommend_similar_courses)]
	# recommend_dissimilar_courses = recommendations(dataframe, inputed_course, cosine_similarity_matrix, False)
	# temporary_dissimilar = dataframe[dataframe['course_name'].isin(recommend_dissimilar_courses)]

	# Displaying top 5 similar and dissimilar courses
	st.write("Top 5 most similar courses")
	st.write(temporary_similar)
	# st.write("Top 5 most dissimilar courses")
	# st.write(temporary_dissimilar)

def preparation_for_cbr(dataframe):

	# content-based filtering

	st.header("Content-based Recommendation")
	st.write("This section is entrusted with the responsibility of"
		" analysing a filtered subset of courses based on the **skills**"
		" a learner is looking to develop. This filter can be adjusted on"
		" the sidebar.")
	st.write("This section also finds courses similar to a selected course"
		" based on Content-based recommendation. The learner can choose"
		" any course that has been filtered on the basis of their skills"
		" in the previous section.")
	st.write("Choose course from 'Select Course' dropdown on the sidebar")

	#st.sidebar.header("Filter on Preferences")

	# filter by skills
	multiselect_skills_option_data = []
	for i in range(1000):
		multiselect_skills_option_data = multiselect_skills_option_data + dataframe['skills'][i]
	multiselect_skills_option_data = list(set(multiselect_skills_option_data))
	skills_selected = st.sidebar.multiselect("Select desired skills", multiselect_skills_option_data)

	# buttons used to make the update of filtering
	filtered_courses = None
	courses = None
	inputed_course = "Nothing"


	#if st.sidebar.button("Filter Courses"):
	temp = filter(dataframe, skills_selected, 'skills', 'course_url')

	#st.write(dataframe['course_url'].isin(temp))
	#st.write(dataframe[dataframe['course_url'].isin(temp)])

	filtered_courses = dataframe[dataframe['course_url'].isin(temp)].reset_index()
	st.write("### Filtered courses based on skill preferences")
	filtered_sorted_ranking=filtered_courses.sort_values('weighted_average',ascending=False)
	st.write(filtered_courses)

	#update filtered courses
	#'courses' variable will only contain names of courses
	courses = filtered_courses['course_name']

	#some relevant details
	st.write("**Number of programmes filtered:**",filtered_courses.shape[0])
	st.write("**Number of courses:**",
		filtered_courses[filtered_courses['learning_product_type']=='COURSE'].shape[0])
	st.write("**Number of professional degrees:**",
		filtered_courses[filtered_courses['learning_product_type']=='PROFESSIONAL CERTIFICATE'].shape[0])
	st.write("**Number of specializations:**",
		filtered_courses[filtered_courses['learning_product_type']=='SPECIALIZATION'].shape[0])
	
	#Some basic plots to provide clarity about the filtered courses
	chart = alt.Chart(filtered_courses).mark_bar().encode(
		y = 'course_provided_by:N',
		x = 'count(course_provided_by):Q'
	).properties(
		title = 'Organizations providing these courses'
	)
	st.altair_chart(chart)

	weighted_average_plot=filtered_sorted_ranking.sort_values('weighted_average',ascending=False)
	
	# plt.rc("font", size=20)
	# plt.figure(figsize=(12,6))
	# axis1=sns.barplot(x=weighted_average_plot['weighted_average'].head(10), y=weighted_average_plot['course_name'].head(10), data=weighted_average_plot)
	# plt.xlim(4, 10)
	# plt.title('Best courses by average weights', weight='bold')
	# plt.xlabel('Weighted Average Score', weight='bold')
	# plt.ylabel('course_name', weight='bold')
	# fig=plt.savefig('best_courses.png')
	# st.pyplot(fig)

	fig, ax = plt.subplots(figsize=(12,6))
	sns.barplot(x=weighted_average_plot['weighted_average'].head(10), y=weighted_average_plot['course_name'].head(10), data=weighted_average_plot)	
	ax.set_xlim(4, 10)
	ax.set_title('Best courses by average weights', weight='bold')
	ax.set_xlabel('Weighted Average Score', weight='bold')
	ax.set_ylabel('course_name', weight='bold')
	st.pyplot(fig)	

	# ratings_bar_chart=pd.DataFrame(filtered_courses['weighted_average'].value_counts()).head(50)
	# st.bar_chart(ratings_bar_chart)

	inputed_course = st.sidebar.selectbox("Select the course you are interested in", courses, key='courses')
	# use button to initiate content-based recommendations
	#else:
		#st.write("```Adjust the 'Select Skills' filter on the sidebar```")

	recommend_similar_courses_radio_button = st.sidebar.radio("Recommend Similar Courses", ('no', 'yes'), index=0)
	if (recommend_similar_courses_radio_button=='yes'):
		content_based_recommendations(dataframe, inputed_course, courses)

	# recommend based on selected course

def main():

	#st.sidebar used to edit details in the sidebar
	#if only using st is used it makes changes in the main content of the webpage 

	st.title("Learn Recommendation Engine")
	st.write("Exploring learnings for learning")

	st.header("About the Project")
	st.write("This course recommendation system is a minimalistic system built to help learners"
		" navigate through the courses on Coursera and aided by a"
		" data-driven strategy. A learner could visualize different"
		" features provided in the dataset or interact with this app"
		" to find suitable courses to take. It also can help"
		" identify suitable courses for a learner based on their"
		" learning preferences.")

	# for loading and displaying data
	dataframe = load_data()

	#to calculate weighted average, more the weighted average - better it becomes for recommendation
	v=dataframe['course_rated_by']
	R=dataframe['course_rating']
	C=dataframe['course_rating'].mean()
	m=dataframe['course_rated_by'].quantile(0.70)
	dataframe['weighted_average']=((R*v)+ (C*m))/(v+m)

	st.header("Datasets Used")
	st.write("For the purpose of building this course recommendation system, data from Coursera"
		" was scraped using the requests and beautifulsoup4 libraries."
		" The final dataset thus acquired consists of 1000 instances"
		" and 14 features.")

	st.sidebar.title("Give your Preferences")
	#st.sidebar.header("Tick the checkbox to see Dataset")

	st.markdown("Toggle the **Display Dataset** checkbox on the sidebar"
		" to show or hide the dataset.")

	# toggle button to display dataset
	if st.sidebar.checkbox("Display Dataset", key='disp_data'):
		st.write(dataframe)

		#to show course ratings in the dataframe in the form of a barchart
		# ratings_bar_chart=pd.DataFrame(dataframe['weighted_average']).head()
		# st.bar_chart(ratings_bar_chart)

		fig, ax = plt.subplots(figsize=(12,6))
		sns.lineplot(y=dataframe['weighted_average'].head(10), x=dataframe['course_name'].head(10), data=dataframe)	
		ax.set_xlim(4, 10)
		ax.set_title('Line representation of courses by average weights', weight='bold')
		ax.set_ylabel('Weighted Average Score', weight='bold')
		ax.set_xlabel('course_name', weight='bold')
		st.pyplot(fig)	
	else:
		pass

	st.markdown("### Columns of Dataset with the description "
		" of data contained in them are -")
	st.write("**course_url:** URL to each course's homepage")
	st.write("**course_name:** Names of the courses")
	st.write("**learning_product_type:** Tells whether it is a course,"
		" a professional certificate or a specialization")
	st.write("**course_provided_by:** Institution, Company or Organization "
		" who has provided the course")
	st.write("**course_rating:** Overall rating of the course")
	st.write("**course_rated_by:** Number of learners who rated the course")
	st.write("**enrolled_student_count:** Number of learners enrolled")
	st.write("**course_difficulty:** Difficulty level of the course")
	st.write("**skills:** Relevant skills the course will deal with")
	st.write("**description:** About the contents of the course")
	st.write("**percentage_of_new_career_starts:** Number of learners who started "
		" a new career after taking this course")
	st.write("**percentage_of_pay_increase_or_promotion:** Number of learners who "
		" received a pay increase or promotion after taking this course")
	st.write("**estimated_time_to_complete:** Approximate time to complete that "
		" particular course")
	st.write("**instructors:** Name of the Instructors of the course")

	# initiate Content Based Recommendation(CBR)
	preparation_for_cbr(dataframe)
	
	
if __name__=="__main__":
	main()

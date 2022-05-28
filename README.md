# Course Recommendation Engine :mortar_board:


## About the Project
This project is a simple recommendation engine solely made for the purpose of helping learners explore and learn about fields of their interest by recommending them suitable courses based on their learning preferences. This recommendation engine uses a data-driven strategy to recommend similar courses. 

For filtering courses according to the **skill preferences** provided by learners, courses related to similar skills are recommended and then filtered courses are displayed in descending order of their **weighted average** values. For recommending courses similar to a selected course the concept of **cosine similarity** has been used.

This project is in the form of a web-app and is deployed using [Streamlit](https://www.streamlit.io/).

## Instructions to run the Project on your local system

1. Download the zip file or clone this repository into your system
2. Open the 'Course_Recommendation_Engine' folder and make sure that you are in the directory where 'requirements.txt' and 'recommender.py' files are present.
3. Open that directory in your terminal(command prompt). You can also navigate using 'cd' command to the folder where 'requirements.txt' and 'recommender.py' files are present from the terminal itself.
4. Run the following command while you are in the directory where 'requirements.txt' file is present to install necessary libraries for this course recommendation system to run.
  ```
  pip install -r requirements.txt
  ```
5. Run the streamlit app using the following command while you are in the directory where 'recommender.py' file is present.
  ```
  streamlit run recommender.py
  ```
6. The app should open at http://localhost:8501 in your browser.

## Datasets Used for making this Project
Data from Coursera was scraped by another github user using the requests and beautifulsoup4 libraries. The same datasets have been utilized for making this recommendation engine. The ```scraper.py``` file contains code for scraping data from [https://www.coursera.org/courses](https://www.coursera.org/courses) and generates [coursera-courses-overview.csv](https://github.com/raghav0303/Course_Recommendation_Engine/blob/main/Datasets/coursera-courses-overview.csv). The ```course_scraper.py``` file contains code to scrape details of each individual course and the output is [coursera-individual-courses.csv](https://github.com/raghav0303/Course_Recommendation_Engine/blob/main/Datasets/coursera-individual-courses.csv).  

Both these above datasets have been combined to give [coursera-courses.csv](https://github.com/raghav0303/Course_Recommendation_Engine/blob/main/Datasets/coursera-courses.csv). This file consists of 1000 instances and 14 features and has a size of 1.41 MB.

## A

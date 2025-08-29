An .ipynb notebook is created with the analysis of the data given to us.
Necessary charts and plots are created.
ONe time run -
ingest.py has been run by using python ingest.py and data has been loaded to mongodb cluster with vector search index active.
-- encode_credentials.py is used to encode the username and password
-- app.py has the streamlit code running, use it directly by writing "streamlit run app.py"
-- or spinning up the "docker-compose build --up" 
-- you must have docker daemon installed or docker engine on your system or docker desktop would do too
-- A list of questions is below which has been tested out - 
    Questions - 
What trials are there for multiple myeloma?
Count the number of trials focused on breast cancer.
How many trials are targeting PD-1?
How many trials have a 'COMPLETED' status?
Are there any trials that investigate both Tislelizumab and a small molecule?
Which trials are for CAR-T cell therapies?
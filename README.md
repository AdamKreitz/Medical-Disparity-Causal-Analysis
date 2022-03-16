## Quality of Life Causal Analysis Project

In this project, we aim to establish causality between various socioeconomic variables and life expectancy outcomes in  roughly 166 different countries, noting the strongest connections between economic and political factors with the length of life expectancy. 

### How to Use

To run our project just run the following two lines of code in a Unix shell, we utilized Unbuntu 20.04 LTS but it should work for others.
```
docker pull mglevitt/world_happiness_project:run_project
docker run mglevitt/world_happiness_project:run_project
```
The pull should not take too long, but to run the code may take upwards of half an hour as a result of the many computations being made with PC. The end output should be a dictionary of the relations we found that will be printed to your terminal and all the graphs of the relations we found will pop up on your device as the code is run. Feel free to consult us if you run into any difficulties with getting our code to run properly.

#### Run PC on Your Own Data

The pipeline we developed is flexible to work with any data in the correct format to find causal relations present in the data. You can follow the step by step instructions below to run PC on your own datasets. 

1. In a local terminal, navigate to where you would like to place the repository of our code
2. Clone this repository on to your local machine in the destination of your choice and navigate into the repository of our code with: 
```
git clone https://github.com/mglevitt/Medical-Disparity-Causal-Analysis.git
cd .\Medical-Disparity-Causal-Analysis\
```
3. Install pipenv on your local machine and use it to to install the dependencies needed to run our code with: 
```
pip install pipenv
pipenv install
```
4. Save the data you would like to reformat to src/data as a csv or xlsx file. 
5. When adding in your own datasets there are a few prior cleaning steps that may have to be done. For any of our provided datasets that you want to use, this step can be skipped. First, the data must be global time series data identified by country names or country name and year. All datasets must have a column with country names that has "country" in the column name. The country names also must follow the same naming conventions as other data that you are merging your data with. If you wish to have an included column with years, then "year" must be in the column's name. When combining datasets with a year column, the pipeline will only include years that are in both datasets, so make sure the years are overlapping in their span. 
6. Open the code of src/scripts/download_data.py with whatever method works from you local terminal or file exploror. Edit the 2 variables after the line "# Add your own data here" towards the bottom of the script to have the correct names for your file names in the list for variable name "new_file_names", including the extension .csv or .xlsx, and the name for your output file before .csv. Once these variable are edited, delete the "#" from before the last line of code then to save the changes to this file.
7. Run the code to have your new data file added to src/final_data with your inputted file name with: 
```
pipenv run .\src\scripts\download_data.py
```
8. After all these steps, a table of relations with the most common relations at the top should be outputted in your terminal. 
9. This step is not neccesary, but for further exploration into the causal relations present you can adjust the signifigance level of PC. Higher signifigance will lead to more relations being present and vice versa. The default signifigance is .2 and this value must be between 0 and 1. The signifigance can be adjusted by editing the value of alpha in the last line of code. 

# World Health Causal Analysis

With this project, we analyzed the causal relationships between multiple factors present across most major countries based on the 21st century. 

To run the project just run the following two lines of code in a Unix shell, we utilized Unbuntu 20.04 LTS but it should work for others.

1. docker pull mglevitt/world_happiness_project:run_project
2. docker run mglevitt/world_happiness_project:run_project

The pull should not take too long, but to run the code may take upwards of half an hour as a result of the many computations being made with PC. The end output should be a dictionary of the relations we found that will be printed to your terminal and all the graphs of the relations we found will pop up on your device as the code is run. Feel free to consult us if you run into any difficulties with getting our code to run properly.

## Reformat and combine your own data with World Health Report data
1. In a local terminal, navigate to where you would like to place the repository of our code
2. Clone this repository on to your local machine in the destination of your choice with: 
```
git clone https://github.com/mglevitt/Medical-Disparity-Causal-Analysis.git
```
4. Navigate into the repository of our code with: cd .\Medical-Disparity-Causal-Analysis\ 
5. Install pipenv on your local machine if not previously installed to install the dependencies needed to run our code with: pip install pipenv
6. Install all dependencies on to your local machine with: pipenv install .
7. Save the data you would like to reformat to src/data as a csv or xlsx file
8. Open the code of src/scripts/downloa_data.py with whatever method works from you local terminal or file exploror.
9. Edit the 4 variables after the line "# Add your own data here" towards the bottom of the script to have the correct names for your file name, including the extension .csv or .xlsx, exact name for year and country columns in your data, the name for your output file before .csv, and finally delete the "#" from before the last line of code. 
10. Save the changes to this file.
11. Run the code to have your new data file added to src/final_data with your inputted file name with: pipenv run .\src\scripts\download_data.py

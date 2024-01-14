# Credit scoring dashboard 

## Overview
The project presents the Credit Score dashboard created and hosted with Streamlit that can ve viewed at [this url](https://credit-scoring-dashboard-vo4t4kjrntvwp3fabtzsss.streamlit.app/).
The dashboard's main feature is to display, explain and simulate the client's probability to default for the current loan application.
The dashboard uses the API developed with Flask (/app) that uses the LightGBM model to make the predictions.

Different models have been tested and evaluated, the results are available in [training notebook](01_Training.ipynb).

For more details on training methodology please refer to the [Methodology Note](Methodology%20note.pdf).

## Dashboard specifications
1. Calculate and display the customer credit risk probability score by providing the customer's id.
2. Display the Feature importance analysis.
3. Display the client's profile by providing different kinds of plots.
4. Simulate the client's score by modifying parameters in the client's profile.
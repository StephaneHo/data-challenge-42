# Regenerating test results

Due to a local GPU memory issue on my computer, the test results has been split in two parts. 
Launch in sequence:
- `notebooks/DataChallenge_Test1.ipynb`
- `notebooks/DataChallenge_Test2.ipynb`

At the end, the complete `test_predictions.csv` is assembled and saved.

# Use the pipeline in validation

The notebook `notebooks/DataChallenge_Val.ipynb` allows to use the pipeline on a single random image to check visually what happens. It presents also the code used to get optimized coefficients and validation results.
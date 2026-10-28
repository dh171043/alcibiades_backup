import pandas as pd

df = pd.read_excel("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task1/task1_dtypes.xlsx")
output_file = "output.txt"

with open(output_file, 'a') as file:
    for index, row in df.iterrows():
        state = str(row['Variable'])
        dtype = str(row['DTYPE'])
        printstr = f"The variable {state} is a {dtype} variable."
        file.write(printstr + '\n')
    print(f"\nPrinted {len(df)} rows.")

# next step is to print message to a .txt file and then copy to word doc
# The variable {0} is {1}
import pandas as pd

test = pd.read_csv('test.csv')
length_test = len(test['Image_File'])
testDF = pd.read_csv('test.csv')

for i in range(length_test):
    if (i < 3769):
        testDF['Class'][i] = "Small"
    else:
        testDF['Class'][i] = "Large"


    print(i)
print(testDF)
testDF.to_csv('finalTest.csv')


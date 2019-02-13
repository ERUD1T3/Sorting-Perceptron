#!/home/th3rudite/anaconda3/bin/python
import random
#import torch


def generateTenDigitList():
    '''
        generate a list of ten number
    '''
    random.seed()
    List = []
    #stochastically generate a 10 numbers between 0 and 9, 
    #adds them to List
    for i in range(10): List.append(random.randint(0, 9)) 

    # print(List)
    return List

def generateDataset2(size):
    '''
        Generate data into 2 file, inputfile.txt  and outputfile.txt
    '''
    infile = open('inputdata.txt', 'a')
    outfile = open('outputdata.txt', 'a')

    for i in range(size + 1):
        Unsorted = generateTenDigitList() #unsorted generated list
        Sorted = Unsorted.copy() #copy the unsorted
        Sorted.sort() #sort the copy
        infile.write(str(Unsorted)) #write the list in file
        infile.write("\n") 
        # file.write(" ; ") #delimited between sorted and unsorted
        outfile.write(str(Sorted)) # write sorted to the file
        outfile.write("\n") 
        #print(Unsorted)
        #print(Sorted)
        #file.closed
        infile.close()
        outfile.close()

def generateDataset(size):
    '''
        Generate data to output file
    '''
    with open("data.txt", "a") as file:
        for i in range(size + 1):

            Unsorted = generateTenDigitList() #unsorted generated list
            Sorted = Unsorted.copy() #copy the unsorted
            Sorted.sort() #sort the copy

            file.write(str(Unsorted)) #write the list in file
            # file.write("\n") 
            file.write(" ; ") #delimited between sorted and unsorted
            file.write(str(Sorted)) # write sorted to the file
            file.write("\n") 
            # print(Unsorted)
            # print(Sorted)
            # file.closed


generateDataset2(1000)

